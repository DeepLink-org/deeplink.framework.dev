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
def context_attention(query: Tensor, key: Tensor, value: Tensor, seqlen: Tensor, mask: Tensor) -> Tensor:
    ...

@context_attention.impl_abstract()
def context_attention_abstract(query, key, value, seqlen, mask):
    return query


@context_attention.impl(['cpu', 'cuda'])
def context_attention_impl(query, key, value, seqlen, mask):
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
def paged_attention_decode(query: Tensor, key_cache: Tensor, value_cache: Tensor, block_table: Tensor, context_len: Tensor, maks: Tensor) -> Tensor:
    ...

@paged_attention_decode.impl_abstract()
def paged_attention_decode_abstract(query, key_cache, value_cache, block_table, context_len, mask):
    return query


@paged_attention_decode.impl(['cpu', 'cuda'])
def paged_attention_decode_impl(query, key_cache, value_cache, block_table, context_len, mask):
    return query

@torch._custom_op.impl.custom_op('atb::add_rms_norm')
def add_rms_norm(x1: Tensor, x2: Tensor, gamma: Tensor, epsilon: float) -> tuple[Tensor, Tensor, Tensor]:
    ...

@add_rms_norm.impl_abstract()
def add_rms_norm_abstract(x1, x2, gamma, epsilon):
    return x1 + x2, x1 + x2, x1 + x2


@add_rms_norm.impl(['cpu', 'cuda'])
def add_rms_norm_impl(x1, x2, gamma, epsilon):
    return x1 + x2, x1 + x2, x1 + x2
