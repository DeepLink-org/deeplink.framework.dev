import math

import torch
import torch.nn.functional as F

from torch import Tensor
from typing import Sequence

torch._dynamo.config.suppress_errors = False

# atb mm
@torch._custom_op.impl.custom_op('atb::linear')
def linear(a: Tensor, b: Tensor, bias: Tensor, trans_a: bool, trans_b: bool) -> Tensor:
    ...


@linear.impl_abstract()
def atb_linear_abstract(a, b, bias, trans_a, trans_b):
    if trans_a:
        a = a.t()
    if trans_b:
        b = b.t()
    return torch.matmul(a, b)


@linear.impl(['cpu', 'cuda'])
def atb_linear_impl(a, b, bias, trans_a, trans_b):
    if trans_a:
        a = a.t()
    if trans_b:
        b = b.t()
    out = torch.matmul(a, b)
    if bias:
        out = out + bias
    return out


# atb mm
@torch._custom_op.impl.custom_op('atb::add')
def add(a: Tensor, b: Tensor) -> Tensor:
    ...


@add.impl_abstract()
def add_abstract(a, b,):
    return a + b


@add.impl(['cpu', 'cuda'])
def add_impl(a, b, bias, trans_a, trans_b):
    return a + b


# atb mm
@torch._custom_op.impl.custom_op('atb::bmm')
def bmm(a: Tensor, b: Tensor) -> Tensor:
    ...


@bmm.impl_abstract()
def bmm_abstract(a, b,):
    return a @ b


@bmm.impl(['cpu', 'cuda'])
def bmm_impl(a, b, bias, trans_a, trans_b):
    return a @ b


# atb fused_mm_mm_add
@torch._custom_op.impl.custom_op('atb::fused_mm_mm_add')
def fused_mm_mm_add(a: Tensor, b: Tensor, c: Tensor, d: Tensor) -> Tensor:
    ...


@fused_mm_mm_add.impl_abstract()
def fused_mm_mm_add_abstract(a, b, c, d):
    return torch.mm(a, b) + torch.mm(c, d)


@fused_mm_mm_add.impl(['cpu', 'cuda'])
def fused_mm_mm_add_impl(a, b, c, d):
    return torch.mm(a, b) + torch.mm(c, d)

# atb fused_mm_mm_add
@torch._custom_op.impl.custom_op('atb::rope')
def rope(query: Tensor, key: Tensor, cos: Tensor, sin: Tensor, seqlen: Tensor) -> tuple[Tensor, Tensor]:
    ...

@rope.impl_abstract()
def rope_abstract(query, key, cos, sin, seqlen):
    return query, key


@rope.impl(['cpu', 'cuda'])
def rope_impl(query, key, cos, sin, seqlen):
    return query, key


@torch._custom_op.impl.custom_op('atb::context_attention')
def context_attention(query: Tensor, key: Tensor, value: Tensor, key_cache: Tensor, value_cache: Tensor, seqlen: Tensor, mask: Tensor, num_q_heads: int, num_kv_heads: int) -> Tensor:
    ...

@context_attention.impl_abstract()
def context_attention_abstract(query, key, value, key_cache: Tensor, value_cache: Tensor, seqlen, mask, num_q_heads: int, num_kv_heads: int):
    return query


@context_attention.impl(['cpu', 'cuda'])
def context_attention_impl(query, key, value, key_cache, value_cache, seqlen, mask, num_q_heads: int, num_kv_heads: int):
    return query


@torch._custom_op.impl.custom_op('atb::fill_kv_cache')
def fill_kv_cache(key: Tensor, value: Tensor, key_cache: Tensor, value_cache: Tensor, kv_indices: Tensor) -> tuple[Tensor, Tensor]:
    ...

@fill_kv_cache.impl_abstract()
def fill_kv_cache_abstract(key, value, key_cache, value_cache, kv_indices):
    return key_cache, value_cache


@fill_kv_cache.impl(['cpu', 'cuda'])
def fill_kv_cache_impl(key, value, key_cache, value_cache, kv_indices):
    return key_cache, value_cache

@torch._custom_op.impl.custom_op('atb::paged_attention_decode')
def paged_attention_decode(query: Tensor, key_cache: Tensor, value_cache: Tensor, block_table: Tensor, context_len: Tensor, maks: Tensor, num_q_heads: int, num_kv_heads: int) -> Tensor:
    ...

@paged_attention_decode.impl_abstract()
def paged_attention_decode_abstract(query, key_cache, value_cache, block_table, context_len, mask, num_q_heads: int, num_kv_heads: int):
    return query


@paged_attention_decode.impl(['cpu', 'cuda'])
def paged_attention_decode_impl(query, key_cache, value_cache, block_table, context_len, mask, num_q_heads: int, num_kv_heads: int):
    return query

@torch._custom_op.impl.custom_op('atb::add_rms_norm')
def add_rms_norm(x1: Tensor, x2: Tensor, gamma: Tensor, epsilon: float) -> tuple[Tensor, Tensor]:
    ...

@add_rms_norm.impl_abstract()
def add_rms_norm_abstract(x1, x2, gamma, epsilon):
    return x1 + x2, x1 + x2


@add_rms_norm.impl(['cpu', 'cuda'])
def add_rms_norm_impl(x1, x2, gamma, epsilon):
    return x1 + x2, x1 + x2


@torch._custom_op.impl.custom_op('atb::mlp_gate_v2')
def mlp_gate_v2(input: Tensor, up: Tensor, gate: Tensor, down: Tensor) -> Tensor:
    ...

@mlp_gate_v2.impl_abstract()
def amlp_gate_v2_abstract(input , up, gate, down):
    return input


@mlp_gate_v2.impl(['cpu', 'cuda'])
def mlp_gate_v2_impl(input, up, gate, down):
    return input


@torch._custom_op.impl.custom_op('atb::silu_and_mul')
def silu_and_mul(gate_up: Tensor) -> Tensor:
    ...

@silu_and_mul.impl_abstract()
def silu_and_mul_abstract(gate_up):
    gate, up = torch.tensor_split(gate_up, 2, dim=-1)
    return gate


@silu_and_mul.impl(['cpu', 'cuda'])
def silu_and_mul_impl(gate_up):
    gate, up = torch.tensor_split(gate_up, 2, dim=-1)
    return gate


@torch._custom_op.impl.custom_op('atb::mlp_gate')
def mlp_gate(input: Tensor, gate_up: Tensor, down: Tensor) -> Tensor:
    ...

@mlp_gate.impl_abstract()
def mlp_gate_abstract(input , gate_up, down):
    return input


@mlp_gate.impl(['cpu', 'cuda'])
def mlp_gate_impl(input, gate_up, down):
    return input


@torch._custom_op.impl.custom_op('atb::atb_context_attention')
def atb_context_attention(query: Tensor,
                          key: Tensor,
                          value: Tensor,
                          k_cache: Tensor,
                          v_cache: Tensor,
                          kv_start_indices_1d: Tensor,
                          kv_seqlens_int: Tensor,
                          mask: Tensor,
                          num_heads: int,
                          num_kv_heads: int,
                          kv_head_size: int,
                          block_size: int) -> Tensor:
    ...

@atb_context_attention.impl_abstract()
def atb_context_attention_abstract(query: Tensor,
                                   key: Tensor,
                                   value: Tensor,
                                   k_cache: Tensor,
                                   v_cache: Tensor,
                                   kv_start_indices_1d: Tensor,
                                   kv_seqlens_int: Tensor,
                                   mask: Tensor,
                                   num_heads: int,
                                   num_kv_heads: int,
                                   kv_head_size: int,
                                   block_size: int):
    return query

@atb_context_attention.impl(['cpu', 'cuda'])
def atb_context_attention_impl(query: Tensor,
                               key: Tensor,
                               value: Tensor,
                               k_cache: Tensor,
                               v_cache: Tensor,
                               kv_start_indices_1d: Tensor,
                               kv_seqlens_int: Tensor,
                               mask: Tensor,
                               num_heads: int,
                               num_kv_heads: int,
                               kv_head_size: int,
                               block_size: int):
    return query



@torch._custom_op.impl.custom_op('atb::atb_paged_attention')
def atb_paged_attention(query: Tensor,
                          key: Tensor,
                          value: Tensor,
                          k_cache: Tensor,
                          v_cache: Tensor,
                          kv_start_indices_1d: Tensor,
                          kv_seqlens_int: Tensor,
                          mask: Tensor,
                          block_offset: Tensor,
                          num_heads: int,
                          num_kv_heads: int,
                          kv_head_size: int,
                          block_size: int) -> Tensor:
    ...

@atb_paged_attention.impl_abstract()
def atb_paged_attention_abstract(query: Tensor,
                                   key: Tensor,
                                   value: Tensor,
                                   k_cache: Tensor,
                                   v_cache: Tensor,
                                   kv_start_indices_1d: Tensor,
                                   kv_seqlens_int: Tensor,
                                   mask: Tensor,
                                   block_offset: Tensor,
                                   num_heads: int,
                                   num_kv_heads: int,
                                   kv_head_size: int,
                                   block_size: int):
    return query

@atb_paged_attention.impl(['cpu', 'cuda'])
def atb_paged_attention_impl(query: Tensor,
                               key: Tensor,
                               value: Tensor,
                               k_cache: Tensor,
                               v_cache: Tensor,
                               kv_start_indices_1d: Tensor,
                               kv_seqlens_int: Tensor,
                               mask: Tensor,
                               block_offset: Tensor,
                               num_heads: int,
                               num_kv_heads: int,
                               kv_head_size: int,
                               block_size: int):
    return query

@torch._custom_op.impl.custom_op('atb::add_and_rms_norm')
def add_and_rms_norm(x1: Tensor, x2: Tensor, gamma: Tensor, epsilon: float) -> Tensor:
    ...

@add_and_rms_norm.impl_abstract()
def add_and_rms_norm_abstract(x1, x2, gamma, epsilon):
    return x1 + x2


@add_and_rms_norm.impl(['cpu', 'cuda'])
def add_and_rms_norm_impl(x1, x2, gamma, epsilon):
    return x1 + x2


@torch._custom_op.impl.custom_op('atb::view_and_rope')
def view_and_rope(query: Tensor, key: Tensor, cos: Tensor, sin: Tensor, seqlen: Tensor, view_size: int) -> tuple[Tensor, Tensor]:
    ...

@view_and_rope.impl_abstract()
def view_and_rope_abstract(query: Tensor, key: Tensor, cos: Tensor, sin: Tensor, seqlen: Tensor, view_size: int) -> tuple[Tensor, Tensor]:
    query = torch.ops.aten.view.default(query, [-1, view_size])
    return query


@view_and_rope.impl(['cpu', 'cuda'])
def view_and_rope_impl(query: Tensor, key: Tensor, cos: Tensor, sin: Tensor, seqlen: Tensor, view_size: int) -> tuple[Tensor, Tensor]:
    query = torch.ops.aten.view.default(query, [-1, view_size])
    return query


# context attention
# view
# linear
@torch._custom_op.impl.custom_op('atb::fused_op')
def fused_op(query: Tensor,
                          key: Tensor,
                          value: Tensor,
                          k_cache: Tensor,
                          v_cache: Tensor,
                          kv_start_indices_1d: Tensor,
                          kv_seqlens_int: Tensor,
                          mask: Tensor,
                          num_heads: int,
                          num_kv_heads: int,
                          kv_head_size: int,
                          block_size: int) -> Tensor:
    ...

@fused_op.impl_abstract()
def fused_op_abstract(query: Tensor,
                                   key: Tensor,
                                   value: Tensor,
                                   k_cache: Tensor,
                                   v_cache: Tensor,
                                   kv_start_indices_1d: Tensor,
                                   kv_seqlens_int: Tensor,
                                   mask: Tensor,
                                   num_heads: int,
                                   num_kv_heads: int,
                                   kv_head_size: int,
                                   block_size: int):
    return query

@fused_op.impl(['cpu', 'cuda'])
def fused_op_impl(query: Tensor,
                               key: Tensor,
                               value: Tensor,
                               k_cache: Tensor,
                               v_cache: Tensor,
                               kv_start_indices_1d: Tensor,
                               kv_seqlens_int: Tensor,
                               mask: Tensor,
                               num_heads: int,
                               num_kv_heads: int,
                               kv_head_size: int,
                               block_size: int):
    return query

# rms_norm: 
# view
# linear
@torch._custom_op.impl.custom_op('atb::llama_prefill')
def llama_prefill(hidden_states: Tensor,
                  rms_norm_1_gamma: Tensor,
                  qkv_weight: Tensor,
                  cos: Tensor,
                  sin: Tensor,
                  k_cache: Tensor,
                  v_cache: Tensor,
                  kv_start_indices_1d: Tensor,
                  kv_seqlens_int: Tensor,
                  mask: Tensor,
                  o_weight: Tensor,
                  rms_norm_2_gamma: Tensor,
                  gate_up: Tensor,
                  down: Tensor,
                  eps: float,
                  q_num_heads: int,
                  kv_num_heads: int,
                  head_size: int,
                  block_size: int,) -> Tensor:
    ...

@llama_prefill.impl_abstract()
def llama_prefill_abstract(
                  hidden_states: Tensor,
                  rms_norm_1_gamma: Tensor,
                  qkv_weight: Tensor,
                  cos: Tensor,
                  sin: Tensor,
                  k_cache: Tensor,
                  v_cache: Tensor,
                  kv_start_indices_1d: Tensor,
                  kv_seqlens_int: Tensor,
                  mask: Tensor,
                  o_weight: Tensor,
                  rms_norm_2_gamma: Tensor,
                  gate_up: Tensor,
                  down: Tensor,
                  eps: float,
                  q_num_heads: int,
                  kv_num_heads: int,
                  head_size: int,
                  block_size: int,) -> Tensor:
    return hidden_states

@llama_prefill.impl(['cpu', 'cuda'])
def llama_prefill_impl(
                  hidden_states: Tensor,
                  rms_norm_1_gamma: Tensor,
                  qkv_weight: Tensor,
                  cos: Tensor,
                  sin: Tensor,
                  k_cache: Tensor,
                  v_cache: Tensor,
                  kv_start_indices_1d: Tensor,
                  kv_seqlens_int: Tensor,
                  mask: Tensor,
                  o_weight: Tensor,
                  rms_norm_2_gamma: Tensor,
                  gate_up: Tensor,
                  down: Tensor,
                  eps: float,
                  q_num_heads: int,
                  kv_num_heads: int,
                  head_size: int,
                  block_size: int,) -> Tensor:
    return hidden_states

@torch._custom_op.impl.custom_op('atb::llama_prefill_and_norm')
def llama_prefill_and_norm(hidden_states: Tensor,
                  rms_norm_1_gamma: Tensor,
                  qkv_weight: Tensor,
                  cos: Tensor,
                  sin: Tensor,
                  k_cache: Tensor,
                  v_cache: Tensor,
                  kv_start_indices_1d: Tensor,
                  kv_seqlens_int: Tensor,
                  mask: Tensor,
                  o_weight: Tensor,
                  rms_norm_2_gamma: Tensor,
                  gate_up: Tensor,
                  down: Tensor,
                  rms_norm_3_gamma: Tensor,
                  eps: float,
                  q_num_heads: int,
                  kv_num_heads: int,
                  head_size: int,
                  block_size: int,) -> Tensor:
    ...

@llama_prefill_and_norm.impl_abstract()
def llama_prefill_and_norm_abstract(
                  hidden_states: Tensor,
                  rms_norm_1_gamma: Tensor,
                  qkv_weight: Tensor,
                  cos: Tensor,
                  sin: Tensor,
                  k_cache: Tensor,
                  v_cache: Tensor,
                  kv_start_indices_1d: Tensor,
                  kv_seqlens_int: Tensor,
                  mask: Tensor,
                  o_weight: Tensor,
                  rms_norm_2_gamma: Tensor,
                  gate_up: Tensor,
                  down: Tensor,
                  rms_norm_3_gamma: Tensor,
                  eps: float,
                  q_num_heads: int,
                  kv_num_heads: int,
                  head_size: int,
                  block_size: int,):
    return hidden_states

@llama_prefill_and_norm.impl(['cpu', 'cuda'])
def llama_prefill_and_norm_impl(
                  hidden_states: Tensor,
                  rms_norm_1_gamma: Tensor,
                  qkv_weight: Tensor,
                  cos: Tensor,
                  sin: Tensor,
                  k_cache: Tensor,
                  v_cache: Tensor,
                  kv_start_indices_1d: Tensor,
                  kv_seqlens_int: Tensor,
                  mask: Tensor,
                  o_weight: Tensor,
                  rms_norm_2_gamma: Tensor,
                  gate_up: Tensor,
                  down: Tensor,
                  rms_norm_3_gamma: Tensor,
                  eps: float,
                  q_num_heads: int,
                  kv_num_heads: int,
                  head_size: int,
                  block_size: int,):
    return hidden_states


@torch._custom_op.impl.custom_op('atb::llama_decode')
def llama_decode(hidden_states: Tensor,
                  rms_norm_1_gamma: Tensor,
                  qkv_weight: Tensor,
                  cos: Tensor,
                  sin: Tensor,
                  k_cache: Tensor,
                  v_cache: Tensor,
                  kv_start_indices_1d: Tensor,
                  kv_seqlens_int: Tensor,
                  mask: Tensor,
                  block_offsets: Tensor,
                  o_weight: Tensor,
                  rms_norm_2_gamma: Tensor,
                  gate_up: Tensor,
                  down: Tensor,
                  eps: float,
                  q_num_heads: int,
                  kv_num_heads: int,
                  head_size: int,
                  block_size: int,) -> Tensor:
    ...

@llama_decode.impl_abstract()
def llama_decode_abstract(
                  hidden_states: Tensor,
                  rms_norm_1_gamma: Tensor,
                  qkv_weight: Tensor,
                  cos: Tensor,
                  sin: Tensor,
                  k_cache: Tensor,
                  v_cache: Tensor,
                  kv_start_indices_1d: Tensor,
                  kv_seqlens_int: Tensor,
                  mask: Tensor,
                  block_offsets: Tensor,
                  o_weight: Tensor,
                  rms_norm_2_gamma: Tensor,
                  gate_up: Tensor,
                  down: Tensor,
                  eps: float,
                  q_num_heads: int,
                  kv_num_heads: int,
                  head_size: int,
                  block_size: int,):
    return hidden_states

@llama_decode.impl(['cpu', 'cuda'])
def llama_decode_impl(
                  hidden_states: Tensor,
                  rms_norm_1_gamma: Tensor,
                  qkv_weight: Tensor,
                  cos: Tensor,
                  sin: Tensor,
                  k_cache: Tensor,
                  v_cache: Tensor,
                  kv_start_indices_1d: Tensor,
                  kv_seqlens_int: Tensor,
                  mask: Tensor,
                  block_offsets: Tensor,
                  o_weight: Tensor,
                  rms_norm_2_gamma: Tensor,
                  gate_up: Tensor,
                  down: Tensor,
                  eps: float,
                  q_num_heads: int,
                  kv_num_heads: int,
                  head_size: int,
                  block_size: int,):
    return hidden_states

@torch._custom_op.impl.custom_op('atb::llama_decode_and_norm')
def llama_decode_and_norm(hidden_states: Tensor,
                  rms_norm_1_gamma: Tensor,
                  qkv_weight: Tensor,
                  cos: Tensor,
                  sin: Tensor,
                  k_cache: Tensor,
                  v_cache: Tensor,
                  kv_start_indices_1d: Tensor,
                  kv_seqlens_int: Tensor,
                  mask: Tensor,
                  block_offsets: Tensor,
                  o_weight: Tensor,
                  rms_norm_2_gamma: Tensor,
                  gate_up: Tensor,
                  down: Tensor,
                  rms_norm_3_gamma: Tensor,
                  eps: float,
                  q_num_heads: int,
                  kv_num_heads: int,
                  head_size: int,
                  block_size: int,) -> Tensor:
    ...

@llama_decode_and_norm.impl_abstract()
def llama_decode_and_norm_abstract(
                  hidden_states: Tensor,
                  rms_norm_1_gamma: Tensor,
                  qkv_weight: Tensor,
                  cos: Tensor,
                  sin: Tensor,
                  k_cache: Tensor,
                  v_cache: Tensor,
                  kv_start_indices_1d: Tensor,
                  kv_seqlens_int: Tensor,
                  mask: Tensor,
                  block_offsets: Tensor,
                  o_weight: Tensor,
                  rms_norm_2_gamma: Tensor,
                  gate_up: Tensor,
                  down: Tensor,
                  rms_norm_3_gamma: Tensor,
                  eps: float,
                  q_num_heads: int,
                  kv_num_heads: int,
                  head_size: int,
                  block_size: int,):
    return hidden_states

@llama_decode_and_norm.impl(['cpu', 'cuda'])
def llama_decode_and_norm_impl(
                  hidden_states: Tensor,
                  rms_norm_1_gamma: Tensor,
                  qkv_weight: Tensor,
                  cos: Tensor,
                  sin: Tensor,
                  k_cache: Tensor,
                  v_cache: Tensor,
                  kv_start_indices_1d: Tensor,
                  kv_seqlens_int: Tensor,
                  mask: Tensor,
                  block_offsets: Tensor,
                  o_weight: Tensor,
                  rms_norm_2_gamma: Tensor,
                  gate_up: Tensor,
                  down: Tensor,
                  rms_norm_3_gamma: Tensor,
                  eps: float,
                  q_num_heads: int,
                  kv_num_heads: int,
                  head_size: int,
                  block_size: int,):
    return hidden_states


@torch._custom_op.impl.custom_op('atb::debug')
def debug(
          hidden_states: Tensor,
          residual: Tensor,
          rms_norm_1_gamma: Tensor,
          qkv_weight: Tensor,
          cos: Tensor,
          sin: Tensor,
          k_cache: Tensor,
          v_cache: Tensor,
          kv_start_indices_1d: Tensor,
          kv_seqlens_int: Tensor,
          mask: Tensor,
          o_weight: Tensor,
          rms_norm_2_gamma: Tensor,
          gate_up: Tensor,
          down: Tensor,
          eps_2: float,
          q_num_heads: int,
          kv_num_heads: int,
          head_size: int,
          block_size: int,) -> Tensor:
    ...

@debug.impl_abstract()
def debug_abstract(
          hidden_states: Tensor,
          residual: Tensor,
          rms_norm_1_gamma: Tensor,
          qkv_weight: Tensor,
          cos: Tensor,
          sin: Tensor,
          k_cache: Tensor,
          v_cache: Tensor,
          kv_start_indices_1d: Tensor,
          kv_seqlens_int: Tensor,
          mask: Tensor,
          o_weight: Tensor,
          rms_norm_2_gamma: Tensor,
          gate_up: Tensor,
          down: Tensor,
          eps_2: float,
          q_num_heads: int,
          kv_num_heads: int,
          head_size: int,
          block_size: int,) -> Tensor:
    return hidden_states

@debug.impl(['cpu', 'cuda'])
def debug_impl(
          hidden_states: Tensor,
          residual: Tensor,
          rms_norm_1_gamma: Tensor,
          qkv_weight: Tensor,
          cos: Tensor,
          sin: Tensor,
          k_cache: Tensor,
          v_cache: Tensor,
          kv_start_indices_1d: Tensor,
          kv_seqlens_int: Tensor,
          mask: Tensor,
          o_weight: Tensor,
          rms_norm_2_gamma: Tensor,
          gate_up: Tensor,
          down: Tensor,
          eps_2: float,
          q_num_heads: int,
          kv_num_heads: int,
          head_size: int,
          block_size: int,) -> Tensor:
    return hidden_states
