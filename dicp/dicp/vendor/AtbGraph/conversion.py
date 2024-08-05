import re
import os
import functools
import operator
import _operator
import torch
import math
from typing import (
    Optional,
)
from torch.types import (
    Number,
)
import numpy as np
import sympy
import torch.fx.traceback as fx_traceback
from torch.fx.immutable_collections import immutable_list
from torch._subclasses import FakeTensor
import dicp.vendor.AtbGraph.atb_op as atb_op
import dicp.vendor.AscendGraph.ascend_op as ascend_op
from dicp.dynamo_bridge.utils import symint_in_shape, neg_in_shape, not_all_num_shape, process_sym_name
from dicp.dynamo_bridge.utils import preprocess_expression, find_root_num, merge_disjoint_set
from dicp.vendor.AtbGraph.codegen.utils import (
    get_ascend_dtype
)
from dicp.dynamo_bridge.conversion import register_conversion_impl
from dicp.dynamo_bridge.op_transformer import SingleOpTransformer
from dicp.vendor.AtbGraph import ext_ops

aten = torch.ops.aten
prims = torch.ops.prims
conversions = {}

sd_fp16 = int(os.environ.get("SD_FP16", 0))


def get_reduction_str(r):
    if r == 0:
        return "none"
    elif r == 1:
        return "mean"
    elif r == 2:
        return "sum"
    else:
        raise RuntimeError("not supported yet!")


def try_to_get_dtype(x):
    if isinstance(x, torch.fx.proxy.Proxy):
        if hasattr(x.node, "meta") and "val" in x.node.meta.keys():
            return x.node.meta['val'].dtype
        elif isinstance(x.node.target, ascend_op.Const):
            # handle with const proxy dtype
            assert len(x.node.args) > 1
            return x.node.args[1]
        else:
            return None

    # handle with basic scalar type
    if isinstance(x, bool):
        return torch.bool
    elif isinstance(x, int):
        return torch.int32
    elif isinstance(x, float):
        return torch.float32
    return None


def is_dicp_cpp_support_dtype(dtype):
    if dtype in [torch.float32, torch.float, torch.float16, torch.int32, torch.int64, torch.bool]:
        return True
    return False


def register_conversion(aten_fn):
    """
    Shim to support decorator syntax.
    """
    return functools.partial(
        register_conversion_impl,
        conversions,
        aten_fn,
    )

def add_inplace_operators(num_inplace):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            result = func(self, *args, **kwargs)
            for i in range(num_inplace):
                self.get_proxy(atb_op.Inplace, (result, args[i], i))
            return result
        return wrapper
    return decorator


class AtenToAtbTransformer(SingleOpTransformer):
    def __init__(self, gm):
        super().__init__(gm, conversions)

    @register_conversion(torch.ops.atb.linear.default)
    def linear(self, a, b, bias, trans_a, trans_b):
        return self.get_proxy(atb_op.Linear, (a, b, bias, trans_a, trans_b))

    @register_conversion(torch.ops.atb.add.default)
    def add(self, a, b):
        return self.get_proxy(atb_op.Add, (a, b))

    @register_conversion(torch.ops.atb.fused_mm_mm_add.default)
    def fused_mm_mm_add(self, a, b, c, d):
        mm1 = self.get_proxy(atb_op.Linear, (a, b, None, False, False))
        mm2 = self.get_proxy(atb_op.Linear, (c, d, None, False, False))
        add = self.get_proxy(atb_op.Add, (mm1, mm2))
        graph = self.get_proxy(atb_op.Graph, (mm1, mm2, add), {'output': add})
        return add

    @register_conversion(operator.getitem)
    def identity(self, x, idx):
        return self.get_proxy(atb_op.GetItem, (x, idx))

    @register_conversion(torch.ops.npu.npu_rms_norm.default)
    def npu_rms_norm(self, x, w, eps=1e-6):
        rms_norm = self.get_proxy(atb_op.RmsNorm, (x, w, eps))
        return rms_norm

    @register_conversion(torch.ops.atb.rope.default)
    def rope(self, query, key, cos, sin, seqlen):
        rope = self.get_proxy(atb_op.Rope, (query, key, cos, sin, seqlen))
        inplace_1 = self.get_proxy(atb_op.Inplace, (rope, query, 0))
        inplace_2 = self.get_proxy(atb_op.Inplace, (rope, key, 1))
        return rope

    @register_conversion(torch.ops.atb.context_attention.default)
    def context_attention(self, query, key, value, seqlen, mask):
        import pdb;pdb.set_trace()
        q_head_num = query.node.meta['val'].shape[-2]
        kv_head_num = key.node.meta['val'].shape[-2]
        out = self.get_proxy(atb_op.SelfAttentionPAEncoder, (query, key, value, seqlen, mask, q_head_num, kv_head_num))
        inplace = self.get_proxy(atb_op.Inplace, (out, query))
        return out

    @register_conversion(torch.ops.atb.fill_kv_cache.default)
    def fill_kv_cache(self, key, value, key_cache, value_cache, kv_indices):
        out = self.get_proxy(atb_op.ReshapeAndCache, (key, value, key_cache, value_cache, kv_indices))
        inplace_1 = self.get_proxy(atb_op.Inplace, (out, key_cache, 0))
        inplace_2 = self.get_proxy(atb_op.Inplace, (out, value_cache, 1))
        return out

    @register_conversion(torch.ops.atb.paged_attention_decode.default)
    def paged_attention_decode(self, query, key_cache, value_cache, block_table, context_len, mask):
        q_head_num = query.node.meta['val'].shape[-2]
        kv_head_num = key_cache.node.meta['val'].shape[-2]
        scale = 1. / math.sqrt(query.node.meta['val'].shape[-1])
        out = self.get_proxy(atb_op.PagedAttention, (query, key_cache, value_cache, block_table, context_len, mask, q_head_num, kv_head_num, scale))
        inplace = self.get_proxy(atb_op.Inplace, (out, query))
        return out

    @register_conversion(torch.ops.atb.add_rms_norm.default)
    def add_rms_norm(self, x1, x2, gamma, epsilon):
        out = self.get_proxy(atb_op.AddRmsNorm, (x1, x2, gamma, epsilon))
        return out
