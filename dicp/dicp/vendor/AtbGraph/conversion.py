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

    @register_conversion(torch.ops.infer_ext.rms_norm.default)
    def npu_rms_norm(self, x, w, eps=1e-6):
        rms_norm = self.get_proxy(atb_op.RmsNorm, (x, w, eps))
        return rms_norm

    @register_conversion(torch.ops.atb.rope.default)
    def rope(self, query, key, cos, sin, seqlen):
        # q_shape = list(query.node.meta['val'].shape)
        # need_reshape = False
        # if len(q_shape) == 3:
        #     query = self.get_proxy(atb_op.View, (query, [q_shape[0], q_shape]))
        rope = self.get_proxy(atb_op.Rope, (query, key, cos, sin, seqlen))
        # inplace_1 = self.get_proxy(atb_op.Inplace, (rope, query, 0))
        # inplace_2 = self.get_proxy(atb_op.Inplace, (rope, key, 1))
        return rope

    @register_conversion(torch.ops.atb.context_attention.default)
    def context_attention(self, query, key, value, key_cache, value_cache, seqlen, mask, num_q_heads, num_kv_heads):
        q_head_num = num_q_heads
        kv_head_num = num_kv_heads
        out = self.get_proxy(atb_op.SelfAttentionPAEncoder, (query, key, value, seqlen, mask, q_head_num, kv_head_num))
        # inplace = self.get_proxy(atb_op.Inplace, (out, query))
        return out

    @register_conversion([torch.ops.atb.fill_kv_cache.default, torch.ops.infer_ext.fill_kv_cache.default])
    def fill_kv_cache(self, key, value, key_cache, value_cache, kv_indices):
        out = self.get_proxy(atb_op.ReshapeAndCache, (key, value, key_cache, value_cache, kv_indices))
        inplace_1 = self.get_proxy(atb_op.Inplace, (out, key_cache, 0))
        inplace_2 = self.get_proxy(atb_op.Inplace, (out, value_cache, 1))
        return out

    @register_conversion(torch.ops.atb.paged_attention_decode.default)
    def paged_attention_decode(self, query, key_cache, value_cache, block_table, context_len, mask, num_q_heads, num_kv_heads):
        q_head_num = num_q_heads
        kv_head_num = num_kv_heads
        scale = 1. / math.sqrt(query.node.meta['val'].shape[-1])
        out = self.get_proxy(atb_op.PagedAttention, (query, key_cache, value_cache, block_table, context_len, mask, q_head_num, kv_head_num, scale))
        # inplace = self.get_proxy(atb_op.Inplace, (out, query))
        return out

    @register_conversion(torch.ops.atb.add_rms_norm.default)
    def add_rms_norm(self, x1, x2, gamma, epsilon):
        out = self.get_proxy(atb_op.AddRmsNorm, (x1, x2, gamma, epsilon))
        return out

    @register_conversion(torch.ops.aten.t.default)
    def t(self, input):
        shape = fx_traceback.get_current_meta()['val'].shape
        permute_shape = [i for i in range(len(shape))]
        permute_shape.reverse()
        return self.get_proxy(atb_op.Transpose, (input, permute_shape))

    @register_conversion(torch.ops.aten.mm.default)
    def aten_mm(self, x, y):
        return self.get_proxy(atb_op.Linear, (x, y, None, False, False))

    @register_conversion(torch.ops.aten.add.Tensor)
    def aten_add_tensor(self, x, y):
        return self.get_proxy(atb_op.Add, (x, y))

    @register_conversion(torch.ops.aten.view.default)
    def aten_view(self, x, size):
        return self.get_proxy(atb_op.View, (x, size))

    @register_conversion(torch.ops.aten.split_with_sizes.default)
    def split_with_sizes(self, x, size, dim):
        assert len(size) == 2 or len(size) == 3
        assert len(set(size)) == 1
        split = self.get_proxy(atb_op.SplitSharing, (x, size, dim))
        # graph = self.get_proxy(atb_op.Graph, (split,), {'output': split})
        return split

    @register_conversion(torch.ops.atb.mlp_gate_v2.default)
    def mlp_gate_v2(self, input, up, gate, down):
        # out = self.get_proxy(atb_op.MlpGateV2, (input, up, gate, down))
        # return out
        # input: [batch, seqLen, hiddenSize], half
        # up: [hiddenSize, ffnHiddenSize], half
        # gate: [hiddenSize, ffnHiddenSize], half
        # down: [ffnHiddenSize, hiddenSize], half
        pass


    @register_conversion(torch.ops.atb.silu_and_mul.default)
    def silu_and_mul(self, gate_up):
        split = self.get_proxy(atb_op.SplitSharing, (gate_up, [1, 1], -1))
        gate = self.get_proxy(atb_op.GetItem, (split, 0))
        up = self.get_proxy(atb_op.GetItem, (split, 1))
        act = self.get_proxy(atb_op.Swish, (gate,))
        mul = self.get_proxy(atb_op.Mul, (act, up))
        graph = self.get_proxy(atb_op.Graph, (split, gate, up, act, mul), {'output': mul})
        return mul

    @register_conversion(torch.ops.atb.mlp_gate.default)
    def mlp_gate(self, input, gate_up, down):
        # input: [batch, seqLen, hiddenSize], half
        # gate_up: [ffnHiddenSize * 2, hiddenSize], half
        # down: [hiddenSize, ffnHiddenSize], half
        mm1 = self.get_proxy(atb_op.Linear, (input, gate_up, None, False, True))
        split = self.get_proxy(atb_op.SplitSharing, (mm1, [1, 1], -1))
        gate = self.get_proxy(atb_op.GetItem, (split, 0))
        up = self.get_proxy(atb_op.GetItem, (split, 1))
        act = self.get_proxy(atb_op.Swish, (gate,))
        mul = self.get_proxy(atb_op.Mul, (act, up))
        mm2 = self.get_proxy(atb_op.Linear, (mul, down, None, False, True))
        graph = self.get_proxy(atb_op.Graph, (mm1, split, gate, up, act, mul, mm2), {'output': mm2})
        return mm2

    @register_conversion(torch.ops.infer_ext.add_rms_norm.default)
    def infer_ext_add_rms_norm(self, x1, x2, gamma, epsilon):
        out = self.get_proxy(atb_op.AddRmsNorm, (x1, x2, gamma, epsilon))
        y_out = self.get_proxy(atb_op.GetItem, (out, 0))
        x_out = self.get_proxy(atb_op.GetItem, (out, 2))
        return self.get_proxy(atb_op.Tuple, (x_out, y_out))

    @register_conversion(torch.ops.aten.sym_size)
    def symsize(self, x, dim):
        import pdb;pdb.set_trace()
        pass

    @register_conversion(torch.ops.atb.lmdeploy_llama_context_attention.default)
    def llama_context_attention(self, query,
                                      key,
                                      value,
                                      k_cache,
                                      v_cache,
                                      kv_start_indices_1d,
                                      kv_seqlens_int,
                                      block_size,
                                      num_heads,
                                      num_kv_heads,
                                      kv_head_size):
        k_cache = self.get_proxy(atb_op.View, (k_cache, [-1, block_size, num_kv_heads, kv_head_size]))
        v_cache = self.get_proxy(atb_op.View, (v_cache, [-1, block_size, num_kv_heads, kv_head_size]))
        fill_kv_cache = self.get_proxy(atb_op.ReshapeAndCache, (key, value, k_cache, v_cache, kv_start_indices_1d))
        getitem0 = self.get_proxy(atb_op.GetItem, (fill_kv_cache, 0))
        getitem1 = self.get_proxy(atb_op.GetItem, (fill_kv_cache, 1))
        inplace1 = self.get_proxy(atb_op.Inplace, (fill_kv_cache, k_cache, 0))
        inplace2 = self.get_proxy(atb_op.Inplace, (fill_kv_cache, v_cache, 1))

        out = self.get_proxy(atb_op.SelfAttentionPAEncoder, (query, key, value, kv_seqlens_int, None, num_heads, num_kv_heads))
        inplace3 = self.get_proxy(atb_op.Inplace, (out, query))
        tuple_op = self.get_proxy(atb_op.Tuple, (out, getitem0, getitem1))
        graph = self.get_proxy(atb_op.Graph, (k_cache, v_cache, fill_kv_cache, inplace1, inplace2, out, inplace3), {"output": tuple_op})
        return tuple_op

