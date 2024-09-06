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
        inplace = self.get_proxy(atb_op.Inplace, (out, query))
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
        inplace = self.get_proxy(atb_op.Inplace, (out, query))
        return out

    @register_conversion(torch.ops.atb.add_rms_norm.default)
    def add_rms_norm(self, x1, x2, gamma, epsilon):
        add = self.get_proxy(atb_op.Add, (x1, x2))
        norm = self.get_proxy(atb_op.RmsNorm, (add, gamma, epsilon))
        graph = self.get_proxy(atb_op.Graph, (add, norm), {'output': [norm, add], 'infer_shape': {"type": "equal", "value": [(0, 0), (0, 0)]}})
        return self.get_proxy(atb_op.Tuple, (norm, add))
        # out = self.get_proxy(atb_op.AddRmsNorm, (x1, x2, gamma, epsilon))
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

    @register_conversion(torch.ops.atb.add_and_rms_norm.default)
    def add_and_rms_norm(self, x1, x2, gamma, epsilon):
        add = self.get_proxy(atb_op.Add, (x1, x2))
        norm = self.get_proxy(atb_op.RmsNorm, (add, gamma, epsilon))
        graph = self.get_proxy(atb_op.Graph, (add, norm), {"output": norm})
        return norm

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

    @register_conversion(torch.ops.aten._to_copy.default)
    def to_copy(self, x, dtype=None, layout=None, device=None):
        assert layout is None
        assert device is None
        if dtype is not None:
            return self.get_proxy(atb_op.Cast, (x, dtype))
        raise RuntimeError('not support yet!')

    @register_conversion(torch.ops.aten.sin.default)
    def sin(self, x):
        return self.get_proxy(atb_op.Sin, (x,))

    @register_conversion(torch.ops.aten.cos.default)
    def cos(self, x):
        return self.get_proxy(atb_op.Cos, (x,))

    @register_conversion(torch.ops.aten.cat.default)
    def cat(self, x, dim):
        return self.get_proxy(atb_op.Concat, (x, dim))

    @register_conversion(torch.ops.atb.bmm.default)
    def bmm(self, x1, x2):
        out = self.get_proxy(atb_op.BatchMatMul, (x1, x2))
        return out

    @register_conversion(torch.ops.aten.transpose.int)
    def transpose_int(self, input, dim_1, dim_2):
        shape = fx_traceback.get_current_meta()['val'].shape
        permute_shape = [i for i in range(len(shape))]
        permute_shape[dim_1], permute_shape[dim_2] = permute_shape[dim_2], permute_shape[dim_1]
        return self.get_proxy(atb_op.Transpose, (input, permute_shape))

    @register_conversion(torch.ops.aten.embedding.default)
    def embedding(self, weight, indices, axis):
        return self.get_proxy(atb_op.Gather, (weight, indices, axis))

    @register_conversion(torch.ops.atb.atb_context_attention.default)
    def atb_context_attention(self, query,
                                      key,
                                      value,
                                      k_cache,
                                      v_cache,
                                      kv_start_indices_1d,
                                      kv_seqlens_int,
                                      mask,
                                      num_heads,
                                      num_kv_heads,
                                      kv_head_size,
                                      block_size):
        k_cache = self.get_proxy(atb_op.View, (k_cache, [-1, block_size, num_kv_heads, kv_head_size]))
        v_cache = self.get_proxy(atb_op.View, (v_cache, [-1, block_size, num_kv_heads, kv_head_size]))
        fill_kv_cache = self.get_proxy(atb_op.ReshapeAndCache, (key, value, k_cache, v_cache, kv_start_indices_1d))
        inplace1 = self.get_proxy(atb_op.Inplace, (fill_kv_cache, k_cache, 0))
        inplace2 = self.get_proxy(atb_op.Inplace, (fill_kv_cache, v_cache, 1))
        
        query = self.get_proxy(atb_op.View, (query, [-1, num_heads * kv_head_size]))
        key = self.get_proxy(atb_op.View, (key, [-1, num_heads * kv_head_size]))
        value = self.get_proxy(atb_op.View, (value, [-1, num_heads * kv_head_size]))

        out = self.get_proxy(atb_op.SelfAttentionPAEncoder, (query, key, value, kv_seqlens_int, mask, num_heads, num_kv_heads))
        graph = self.get_proxy(atb_op.Graph, (fill_kv_cache, out), {"output": [out, inplace1, inplace2]})
        return out

    @register_conversion(torch.ops.atb.atb_paged_attention.default)
    def atb_paged_attention(self, query,
                                      key,
                                      value,
                                      k_cache,
                                      v_cache,
                                      kv_start_indices_1d,
                                      kv_seqlens_int,
                                      mask,
                                      block_offset,
                                      num_heads,
                                      num_kv_heads,
                                      kv_head_size,
                                      block_size):
        k_cache = self.get_proxy(atb_op.View, (k_cache, [-1, block_size, num_kv_heads, kv_head_size]))
        v_cache = self.get_proxy(atb_op.View, (v_cache, [-1, block_size, num_kv_heads, kv_head_size]))
        fill_kv_cache = self.get_proxy(atb_op.ReshapeAndCache, (key, value, k_cache, v_cache, kv_start_indices_1d))
        inplace1 = self.get_proxy(atb_op.Inplace, (fill_kv_cache, k_cache, 0))
        inplace2 = self.get_proxy(atb_op.Inplace, (fill_kv_cache, v_cache, 1))
        
        scale = 1. / math.sqrt(kv_head_size)
        out = self.get_proxy(atb_op.PagedAttention, (query, k_cache, v_cache, block_offset, kv_seqlens_int, mask, num_heads, num_kv_heads, scale))
        graph = self.get_proxy(atb_op.Graph, (fill_kv_cache, out), {"output": [out, inplace1, inplace2]})
        return out


    @register_conversion(torch.ops.atb.debug.default)
    def debug(self,
          hidden_states,
          residual,
          rms_norm_1_gamma,
          qkv_weight,
          cos,
          sin,
          k_cache,
          v_cache,
          kv_start_indices_1d,
          kv_seqlens_int,
          mask,
          o_weight,
          rms_norm_2_gamma,
          gate_up,
          down,
          eps,
          q_num_heads,
          kv_num_heads,
          head_size,
          block_size,):

        # if residual is None:
        #     residual = hidden_states
        # else:
        # residual2 = self.get_proxy(atb_op.Add, (hidden_states, residual))
        rms_norm = self.get_proxy(atb_op.RmsNorm, (hidden_states, rms_norm_1_gamma, eps))
        linear = self.get_proxy(atb_op.Linear, (rms_norm, qkv_weight, None, False, True))
        
        all_heads = q_num_heads + kv_num_heads + kv_num_heads
        
        view = self.get_proxy(atb_op.View, (linear, [-1, all_heads, head_size]))
        split = self.get_proxy(atb_op.SplitSharing, (view, [q_num_heads, kv_num_heads, kv_num_heads], 1))
        getitem = self.get_proxy(atb_op.GetItem, (split, 0))
        getitem_1 = self.get_proxy(atb_op.GetItem, (split, 1))
        getitem_2 = self.get_proxy(atb_op.GetItem, (split, 2))

        
        view_1 = self.get_proxy(atb_op.View, (getitem, [-1, q_num_heads * head_size]))
        view_2 = self.get_proxy(atb_op.View, (getitem_1, [-1, kv_num_heads * head_size]))
        rope = self.get_proxy(atb_op.Rope, (view_1, view_2, cos, sin, kv_seqlens_int))
        getitem_3 = self.get_proxy(atb_op.GetItem, (rope, 0))
        getitem_4 = self.get_proxy(atb_op.GetItem, (rope, 1))

        view_3 = self.get_proxy(atb_op.View, (getitem_3, [-1, q_num_heads, head_size]))
        view_4 = self.get_proxy(atb_op.View, (getitem_4, [-1, kv_num_heads, head_size]))
        
        view_5 = self.get_proxy(atb_op.View, (k_cache, [-1, block_size, kv_num_heads, head_size]))
        view_6 = self.get_proxy(atb_op.View, (v_cache, [-1, block_size, kv_num_heads, head_size]))
        fill_kv_cache = self.get_proxy(atb_op.ReshapeAndCache, (view_4, getitem_2, view_5, view_6, kv_start_indices_1d))
        inplace1 = self.get_proxy(atb_op.Inplace, (fill_kv_cache, view_5, 0))
        inplace2 = self.get_proxy(atb_op.Inplace, (fill_kv_cache, view_6, 1))
        
        view_7 = self.get_proxy(atb_op.View, (view_3, [-1, q_num_heads * head_size]))
        view_8 = self.get_proxy(atb_op.View, (view_4, [-1, kv_num_heads * head_size]))
        view_9 = self.get_proxy(atb_op.View, (getitem_2, [-1, kv_num_heads * head_size]))
        context_attention = self.get_proxy(atb_op.SelfAttentionPAEncoder, (view_7, view_8, view_9, kv_seqlens_int, mask, q_num_heads, kv_num_heads))

        view_10 = self.get_proxy(atb_op.View, (context_attention, [1, -1, q_num_heads * head_size]))
        linear_1 = self.get_proxy(atb_op.Linear, (view_10, o_weight, None, False, True))

        add = self.get_proxy(atb_op.Add, (linear_1, hidden_states))
        rms_norm_1 = self.get_proxy(atb_op.RmsNorm, (add, rms_norm_2_gamma, eps))

        # input: rms_norm_1
        mlp_mm1 = self.get_proxy(atb_op.Linear, (rms_norm_1, gate_up, None, False, True))
        mlp_split = self.get_proxy(atb_op.SplitSharing, (mlp_mm1, [1, 1], -1))
        mlp_gate = self.get_proxy(atb_op.GetItem, (mlp_split, 0))
        mlp_up = self.get_proxy(atb_op.GetItem, (mlp_split, 1))
        mlp_act = self.get_proxy(atb_op.Swish, (mlp_gate,))
        mlp_mul = self.get_proxy(atb_op.Mul, (mlp_act, mlp_up))
        mlp = self.get_proxy(atb_op.Linear, (mlp_mul, down, None, False, True))

        add_1 = self.get_proxy(atb_op.Add, (mlp, add))

        graph_0 = self.get_proxy(atb_op.Graph, ( 
                                              rms_norm,
                                              linear,
                                              split,
                                              getitem,
                                              getitem_1,
                                              getitem_2,
                                              rope,
                                              getitem_3,
                                              getitem_4,
                                              fill_kv_cache,
                                              context_attention,
                                              linear_1,
                                              add,
                                              rms_norm_1,
                                              mlp_mm1,
                                              mlp_split,
                                              mlp_gate,
                                              mlp_up,
                                              mlp_act,
                                              mlp_mul,
                                              mlp,
                                              add_1
                                              ), {"output": [add_1, mlp_gate, mlp_up, getitem, getitem_1, getitem_2]})
        return add_1

    @register_conversion(torch.ops.atb.llama_prefill.default)
    def atb_llama_prefill(self,
                          hidden_states,
                          rms_norm_1_gamma,
                          qkv_weight,
                          cos,
                          sin,
                          k_cache,
                          v_cache,
                          kv_start_indices_1d,
                          kv_seqlens_int,
                          mask,
                          o_weight,
                          rms_norm_2_gamma,
                          gate_up,
                          down,
                          eps,
                          q_num_heads,
                          kv_num_heads,
                          head_size,
                          block_size):
        rms_norm = self.get_proxy(atb_op.RmsNorm, (hidden_states, rms_norm_1_gamma, eps))
        linear = self.get_proxy(atb_op.Linear, (rms_norm, qkv_weight, None, False, True))
        
        all_heads = q_num_heads + kv_num_heads + kv_num_heads
        view = self.get_proxy(atb_op.View, (linear, [-1, all_heads, head_size]))
        split = self.get_proxy(atb_op.SplitSharing, (view, [q_num_heads, kv_num_heads, kv_num_heads], 1))
        getitem = self.get_proxy(atb_op.GetItem, (split, 0))
        getitem_1 = self.get_proxy(atb_op.GetItem, (split, 1))
        getitem_2 = self.get_proxy(atb_op.GetItem, (split, 2))
        
        view_1 = self.get_proxy(atb_op.View, (getitem, [-1, q_num_heads * head_size]))
        view_2 = self.get_proxy(atb_op.View, (getitem_1, [-1, kv_num_heads * head_size]))
        rope = self.get_proxy(atb_op.Rope, (view_1, view_2, cos, sin, kv_seqlens_int))
        getitem_3 = self.get_proxy(atb_op.GetItem, (rope, 0))
        getitem_4 = self.get_proxy(atb_op.GetItem, (rope, 1))

        view_3 = self.get_proxy(atb_op.View, (getitem_3, [-1, q_num_heads, head_size]))
        view_4 = self.get_proxy(atb_op.View, (getitem_4, [-1, kv_num_heads, head_size]))
        
        view_5 = self.get_proxy(atb_op.View, (k_cache, [-1, block_size, kv_num_heads, head_size]))
        view_6 = self.get_proxy(atb_op.View, (v_cache, [-1, block_size, kv_num_heads, head_size]))
        fill_kv_cache = self.get_proxy(atb_op.ReshapeAndCache, (view_4, getitem_2, view_5, view_6, kv_start_indices_1d))
        inplace1 = self.get_proxy(atb_op.Inplace, (fill_kv_cache, view_5, 0))
        inplace2 = self.get_proxy(atb_op.Inplace, (fill_kv_cache, view_6, 1))
        
        view_7 = self.get_proxy(atb_op.View, (view_3, [-1, q_num_heads * head_size]))
        view_8 = self.get_proxy(atb_op.View, (view_4, [-1, kv_num_heads * head_size]))
        view_9 = self.get_proxy(atb_op.View, (getitem_2, [-1, kv_num_heads * head_size]))
        context_attention = self.get_proxy(atb_op.SelfAttentionPAEncoder, (view_7, view_8, view_9, kv_seqlens_int, mask, q_num_heads, kv_num_heads))


        view_10 = self.get_proxy(atb_op.View, (context_attention, [1, -1, q_num_heads * head_size]))
        linear_1 = self.get_proxy(atb_op.Linear, (view_10, o_weight, None, False, True))

        add = self.get_proxy(atb_op.Add, (linear_1, hidden_states))
        rms_norm_1 = self.get_proxy(atb_op.RmsNorm, (add, rms_norm_2_gamma, eps))

        # input: rms_norm_1
        mlp_mm1 = self.get_proxy(atb_op.Linear, (rms_norm_1, gate_up, None, False, True))
        mlp_split = self.get_proxy(atb_op.SplitSharing, (mlp_mm1, [1, 1], -1))
        mlp_gate = self.get_proxy(atb_op.GetItem, (mlp_split, 0))
        mlp_up = self.get_proxy(atb_op.GetItem, (mlp_split, 1))
        mlp_act = self.get_proxy(atb_op.Swish, (mlp_gate,))
        mlp_mul = self.get_proxy(atb_op.Mul, (mlp_act, mlp_up))
        mlp = self.get_proxy(atb_op.Linear, (mlp_mul, down, None, False, True))
        add_1 = self.get_proxy(atb_op.Add, (mlp, add))
        graph = self.get_proxy(atb_op.Graph, ( 
                                              rms_norm,
                                              linear,
                                              split,
                                              getitem,
                                              getitem_1,
                                              getitem_2,
                                              rope,
                                              getitem_3,
                                              getitem_4,
                                              fill_kv_cache,
                                              context_attention,
                                              linear_1,
                                              add,
                                              rms_norm_1,
                                              mlp_mm1,
                                              mlp_split,
                                              mlp_gate,
                                              mlp_up,
                                              mlp_act,
                                              mlp_mul,
                                              mlp,
                                              add_1
                                              ), {"output": [add_1,], 'infer_shape': {"type": "equal", "value": [(0, 0)]}})
        return add_1

    @register_conversion(torch.ops.atb.llama_prefill_and_norm.default)
    def atb_llama_prefill_and_norm(self,
                          hidden_states,
                          rms_norm_1_gamma,
                          qkv_weight,
                          cos,
                          sin,
                          k_cache,
                          v_cache,
                          kv_start_indices_1d,
                          kv_seqlens_int,
                          mask,
                          o_weight,
                          rms_norm_2_gamma,
                          gate_up,
                          down,
                          rms_norm_3_gamma,
                          eps,
                          q_num_heads,
                          kv_num_heads,
                          head_size,
                          block_size):
        rms_norm = self.get_proxy(atb_op.RmsNorm, (hidden_states, rms_norm_1_gamma, eps))
        linear = self.get_proxy(atb_op.Linear, (rms_norm, qkv_weight, None, False, True))
        
        all_heads = q_num_heads + kv_num_heads + kv_num_heads
        view = self.get_proxy(atb_op.View, (linear, [-1, all_heads, head_size]))
        split = self.get_proxy(atb_op.SplitSharing, (view, [q_num_heads, kv_num_heads, kv_num_heads], 1))
        getitem = self.get_proxy(atb_op.GetItem, (split, 0))
        getitem_1 = self.get_proxy(atb_op.GetItem, (split, 1))
        getitem_2 = self.get_proxy(atb_op.GetItem, (split, 2))
        
        view_1 = self.get_proxy(atb_op.View, (getitem, [-1, q_num_heads * head_size]))
        view_2 = self.get_proxy(atb_op.View, (getitem_1, [-1, q_num_heads * head_size]))
        rope = self.get_proxy(atb_op.Rope, (view_1, view_2, cos, sin, kv_seqlens_int))
        getitem_3 = self.get_proxy(atb_op.GetItem, (rope, 0))
        getitem_4 = self.get_proxy(atb_op.GetItem, (rope, 1))
        
        view_3 = self.get_proxy(atb_op.View, (getitem_3, [-1, q_num_heads, head_size]))
        view_4 = self.get_proxy(atb_op.View, (getitem_4, [-1, kv_num_heads, head_size]))
        
        view_5 = self.get_proxy(atb_op.View, (k_cache, [-1, block_size, kv_num_heads, head_size]))
        view_6 = self.get_proxy(atb_op.View, (v_cache, [-1, block_size, kv_num_heads, head_size]))
        fill_kv_cache = self.get_proxy(atb_op.ReshapeAndCache, (view_4, getitem_2, view_5, view_6, kv_start_indices_1d))
        inplace1 = self.get_proxy(atb_op.Inplace, (fill_kv_cache, view_5, 0))
        inplace2 = self.get_proxy(atb_op.Inplace, (fill_kv_cache, view_6, 1))
        
        view_7 = self.get_proxy(atb_op.View, (view_3, [-1, q_num_heads * head_size]))
        view_8 = self.get_proxy(atb_op.View, (view_4, [-1, kv_num_heads * head_size]))
        view_9 = self.get_proxy(atb_op.View, (getitem_2, [-1, kv_num_heads * head_size]))
        context_attention = self.get_proxy(atb_op.SelfAttentionPAEncoder, (view_7, view_8, view_9, kv_seqlens_int, mask, q_num_heads, kv_num_heads))

        view_10 = self.get_proxy(atb_op.View, (context_attention, [1, -1, q_num_heads * head_size]))
        linear_1 = self.get_proxy(atb_op.Linear, (view_10, o_weight, None, False, True))

        add = self.get_proxy(atb_op.Add, (linear_1, hidden_states))
        rms_norm_1 = self.get_proxy(atb_op.RmsNorm, (add, rms_norm_2_gamma, eps))

        # input: rms_norm_1
        mlp_mm1 = self.get_proxy(atb_op.Linear, (rms_norm_1, gate_up, None, False, True))
        mlp_split = self.get_proxy(atb_op.SplitSharing, (mlp_mm1, [1, 1], -1))
        mlp_gate = self.get_proxy(atb_op.GetItem, (mlp_split, 0))
        mlp_up = self.get_proxy(atb_op.GetItem, (mlp_split, 1))
        mlp_act = self.get_proxy(atb_op.Swish, (mlp_gate,))
        mlp_mul = self.get_proxy(atb_op.Mul, (mlp_act, mlp_up))
        mlp = self.get_proxy(atb_op.Linear, (mlp_mul, down, None, False, True))

        add_1 = self.get_proxy(atb_op.Add, (mlp, add))
        rms_norm_2 = self.get_proxy(atb_op.RmsNorm, (add_1, rms_norm_3_gamma, eps))

        graph = self.get_proxy(atb_op.Graph, (rms_norm,
                                              linear,
                                              view,
                                              split,
                                              getitem,
                                              getitem,
                                              getitem_1,
                                              getitem_2,
                                              view_1,
                                              view_2,
                                              rope,
                                              getitem_3,
                                              getitem_4,
                                              view_3,
                                              view_4,
                                              view_5,
                                              view_6,
                                              fill_kv_cache,
                                              view_7,
                                              view_8,
                                              view_9,
                                              context_attention,
                                              view_10,
                                              linear_1,
                                              add,
                                              rms_norm_1,
                                              mlp_mm1,
                                              mlp_split,
                                              mlp_gate,
                                              mlp_up,
                                              mlp_act,
                                              mlp_mul,
                                              mlp,
                                              add_1,
                                              rms_norm_2), {"output": [rms_norm_2,], 'infer_shape': {"type": "equal", "value": [(0, 0)]}})
                                            #   rms_norm_2), {"output": rms_norm_2, 'infer_shape': {"type": "equal", "value": [(0, 0)]}})
        return rms_norm_2


    @register_conversion(torch.ops.atb.llama_decode.default)
    def atb_llama_decode(self,
                          hidden_states,
                          rms_norm_1_gamma,
                          qkv_weight,
                          cos,
                          sin,
                          k_cache,
                          v_cache,
                          kv_start_indices_1d,
                          kv_seqlens_int,
                          mask,
                          block_offsets,
                          o_weight,
                          rms_norm_2_gamma,
                          gate_up,
                          down,
                          eps,
                          q_num_heads,
                          kv_num_heads,
                          head_size,
                          block_size):
        rms_norm = self.get_proxy(atb_op.RmsNorm, (hidden_states, rms_norm_1_gamma, eps))
        linear = self.get_proxy(atb_op.Linear, (rms_norm, qkv_weight, None, False, True))
        
        all_heads = q_num_heads + kv_num_heads + kv_num_heads
        view = self.get_proxy(atb_op.View, (linear, [-1, all_heads, head_size]))
        split = self.get_proxy(atb_op.SplitSharing, (view, [q_num_heads, kv_num_heads, kv_num_heads], 1))
        getitem = self.get_proxy(atb_op.GetItem, (split, 0))
        getitem_1 = self.get_proxy(atb_op.GetItem, (split, 1))
        getitem_2 = self.get_proxy(atb_op.GetItem, (split, 2))

        view_1 = self.get_proxy(atb_op.View, (getitem, [-1, q_num_heads * head_size]))
        view_2 = self.get_proxy(atb_op.View, (getitem_1, [-1, kv_num_heads * head_size]))
        rope = self.get_proxy(atb_op.Rope, (view_1, view_2, cos, sin, kv_seqlens_int))
        getitem_3 = self.get_proxy(atb_op.GetItem, (rope, 0))
        getitem_4 = self.get_proxy(atb_op.GetItem, (rope, 1))

        view_3 = self.get_proxy(atb_op.View, (getitem_3, [-1, q_num_heads, head_size]))
        view_4 = self.get_proxy(atb_op.View, (getitem_4, [-1, kv_num_heads, head_size]))
        
        view_5 = self.get_proxy(atb_op.View, (k_cache, [-1, block_size, kv_num_heads, head_size]))
        view_6 = self.get_proxy(atb_op.View, (v_cache, [-1, block_size, kv_num_heads, head_size]))
        fill_kv_cache = self.get_proxy(atb_op.ReshapeAndCache, (view_4, getitem_2, view_5, view_6, kv_start_indices_1d))
        inplace1 = self.get_proxy(atb_op.Inplace, (fill_kv_cache, view_5, 0))
        inplace2 = self.get_proxy(atb_op.Inplace, (fill_kv_cache, view_6, 1))
        
        scale = 1. / math.sqrt(head_size)
        paged_attention = self.get_proxy(atb_op.PagedAttention, (view_3, view_5, view_6, block_offsets, kv_seqlens_int, mask, q_num_heads, kv_num_heads, scale))

        view_10 = self.get_proxy(atb_op.View, (paged_attention, [1, -1, q_num_heads * head_size]))
        linear_1 = self.get_proxy(atb_op.Linear, (view_10, o_weight, None, False, True))

        add = self.get_proxy(atb_op.Add, (linear_1, hidden_states))
        rms_norm_1 = self.get_proxy(atb_op.RmsNorm, (add, rms_norm_2_gamma, eps))


        # input: rms_norm_1
        mlp_mm1 = self.get_proxy(atb_op.Linear, (rms_norm_1, gate_up, None, False, True))
        mlp_split = self.get_proxy(atb_op.SplitSharing, (mlp_mm1, [1, 1], -1))
        mlp_gate = self.get_proxy(atb_op.GetItem, (mlp_split, 0))
        mlp_up = self.get_proxy(atb_op.GetItem, (mlp_split, 1))
        mlp_act = self.get_proxy(atb_op.Swish, (mlp_gate,))
        mlp_mul = self.get_proxy(atb_op.Mul, (mlp_act, mlp_up))
        mlp = self.get_proxy(atb_op.Linear, (mlp_mul, down, None, False, True))
        add_1 = self.get_proxy(atb_op.Add, (mlp, add))

        graph = self.get_proxy(atb_op.Graph, ( 
                                              rms_norm,
                                              linear,
                                              split,
                                              getitem,
                                              getitem_1,
                                              getitem_2,
                                              rope,
                                              getitem_3,
                                              getitem_4,
                                              fill_kv_cache,
                                              paged_attention,
                                              linear_1,
                                              add,
                                              rms_norm_1,
                                              mlp_mm1,
                                              mlp_split,
                                              mlp_gate,
                                              mlp_up,
                                              mlp_act,
                                              mlp_mul,
                                              mlp,
                                              add_1
                                              ), {"output": [
                                                #   linear,
                                                    # paged_attention,
                                                    # getitem,
                                                    # getitem_1,
                                                    # getitem_2,
                                                    # getitem_3,
                                                    # getitem_4,

                                                    # mlp_mm1,
                                                    # mlp_gate,
                                                    # mlp_up,
                                                    # mlp_act,
                                                    # mlp_mul,
                                                    # mlp,
                                                    add_1,
                                                    ], 'infer_shape': {"type": "equal", "value": [(0, 0)]}})



        return add_1

    @register_conversion(torch.ops.atb.llama_decode_and_norm.default)
    def atb_llama_decode_and_norm(self,
                          hidden_states,
                          rms_norm_1_gamma,
                          qkv_weight,
                          cos,
                          sin,
                          k_cache,
                          v_cache,
                          kv_start_indices_1d,
                          kv_seqlens_int,
                          mask,
                          block_offsets,
                          o_weight,
                          rms_norm_2_gamma,
                          gate_up,
                          down,
                          rms_norm_3_gamma,
                          eps,
                          q_num_heads,
                          kv_num_heads,
                          head_size,
                          block_size):
        rms_norm = self.get_proxy(atb_op.RmsNorm, (hidden_states, rms_norm_1_gamma, eps))
        linear = self.get_proxy(atb_op.Linear, (rms_norm, qkv_weight, None, False, True))
        
        all_heads = q_num_heads + kv_num_heads + kv_num_heads
        view = self.get_proxy(atb_op.View, (linear, [-1, all_heads, head_size]))
        split = self.get_proxy(atb_op.SplitSharing, (view, [q_num_heads, kv_num_heads, kv_num_heads], 1))
        getitem = self.get_proxy(atb_op.GetItem, (split, 0))
        getitem_1 = self.get_proxy(atb_op.GetItem, (split, 1))
        getitem_2 = self.get_proxy(atb_op.GetItem, (split, 2))
        
        view_1 = self.get_proxy(atb_op.View, (getitem, [-1, q_num_heads * head_size]))
        view_2 = self.get_proxy(atb_op.View, (getitem_1, [-1, q_num_heads * head_size]))
        rope = self.get_proxy(atb_op.Rope, (view_1, view_2, cos, sin, kv_seqlens_int))
        getitem_3 = self.get_proxy(atb_op.GetItem, (rope, 0))
        getitem_4 = self.get_proxy(atb_op.GetItem, (rope, 1))
        
        view_3 = self.get_proxy(atb_op.View, (getitem_3, [-1, q_num_heads, head_size]))
        view_4 = self.get_proxy(atb_op.View, (getitem_4, [-1, kv_num_heads, head_size]))
        
        view_5 = self.get_proxy(atb_op.View, (k_cache, [-1, block_size, kv_num_heads, head_size]))
        view_6 = self.get_proxy(atb_op.View, (v_cache, [-1, block_size, kv_num_heads, head_size]))
        fill_kv_cache = self.get_proxy(atb_op.ReshapeAndCache, (view_4, getitem_2, view_5, view_6, kv_start_indices_1d))
        inplace1 = self.get_proxy(atb_op.Inplace, (fill_kv_cache, view_5, 0))
        inplace2 = self.get_proxy(atb_op.Inplace, (fill_kv_cache, view_6, 1))
        
        scale = 1. / math.sqrt(head_size)
        paged_attention = self.get_proxy(atb_op.PagedAttention, (view_3, view_5, view_6, block_offsets, kv_seqlens_int, mask, q_num_heads, kv_num_heads, scale))

        view_10 = self.get_proxy(atb_op.View, (paged_attention, [1, -1, q_num_heads * head_size]))
        linear_1 = self.get_proxy(atb_op.Linear, (view_10, o_weight, None, False, True))

        add = self.get_proxy(atb_op.Add, (linear_1, hidden_states))
        rms_norm_1 = self.get_proxy(atb_op.RmsNorm, (add, rms_norm_2_gamma, eps))

        # input: rms_norm_1
        mlp_mm1 = self.get_proxy(atb_op.Linear, (rms_norm_1, gate_up, None, False, True))
        mlp_split = self.get_proxy(atb_op.SplitSharing, (mlp_mm1, [1, 1], -1))
        mlp_gate = self.get_proxy(atb_op.GetItem, (mlp_split, 0))
        mlp_up = self.get_proxy(atb_op.GetItem, (mlp_split, 1))
        mlp_act = self.get_proxy(atb_op.Swish, (mlp_gate,))
        mlp_mul = self.get_proxy(atb_op.Mul, (mlp_act, mlp_up))
        mlp = self.get_proxy(atb_op.Linear, (mlp_mul, down, None, False, True))

        add_1 = self.get_proxy(atb_op.Add, (mlp, add))
        rms_norm_2 = self.get_proxy(atb_op.RmsNorm, (add_1, rms_norm_3_gamma, eps))
        graph = self.get_proxy(atb_op.Graph, (rms_norm,
                                              linear,
                                              view,
                                              split,
                                              getitem,
                                              getitem,
                                              getitem_1,
                                              getitem_2,
                                              rope,
                                              getitem_3,
                                              getitem_4,
                                              fill_kv_cache,
                                              paged_attention,
                                              view_10,
                                              linear_1,
                                              add,
                                              rms_norm_1,
                                              mlp_mm1,
                                              mlp_split,
                                              mlp_gate,
                                              mlp_up,
                                              mlp_act,
                                              mlp_mul,
                                              mlp,
                                              add_1,
                                              rms_norm_2), {"output": [rms_norm_2,], 'infer_shape': {"type": "equal", "value": [(0, 0)]}})
                                            #   rms_norm_2), {"output": rms_norm_2, 'infer_shape': {"type": "equal", "value": [(0, 0)]}})
        return rms_norm_2

    @register_conversion(torch.ops.aten.unsqueeze.default)
    def unsqueeze(self, x, dim):
        return self.get_proxy(atb_op.Unsqueeze, (x, dim))

    @register_conversion(torch.ops.aten.squeeze.dim)
    def squeeze(self, x, dim):
        return self.get_proxy(atb_op.Squeeze, (x, dim))
