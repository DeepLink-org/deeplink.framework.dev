import acl
import torch
from typing import Tuple
from dicp.dynamo_bridge.operator import Operator
from dicp.vendor.AscendGraph.infer_res_utils import *
from dicp.vendor.AscendGraph.codegen.utils import (
    check_ret,
    get_acl_format,
    get_acl_dtype,
    get_shape_from_desc,
    get_torch_dtype
)
from dicp.dynamo_bridge.utils import get_memory_format

aten = torch.ops.aten


def negative_in_shape(shape):
    for elem in shape:
        if elem < 0:
            return True
    return False


class Linear(Operator):
    def __init__(self):
        super().__init__("Linear")
    
    def infer_result(self, a, b, bias, trans_a, trans_b):
        if trans_a:
            a = a.t()
        if trans_b:
            b = b.t()
        out = torch.matmul(a, b)
        if bias:
            out = out + bias
        return out

class Add(Operator):
    def __init__(self):
        super().__init__("Add")
    
    def infer_result(self, a, b):
        return a + b

class Graph(Operator):
    def __init__(self):
        super().__init__("Graph")
        
    def infer_result(self, *args, **kwargs):
        if not isinstance(kwargs['output'], list):
            return kwargs['output'].meta['val']
        else:
            res = [x.meta['val'] for x in kwargs['output']]
            return tuple(res)
 

class GetItem(Operator):
    def __init__(self):
        super().__init__("GetItem")

    def infer_result(self, x, index):
        return x[index]


class RmsNorm(Operator):
    def __init__(self,):
        super().__init__("RmsNorm")
    
    def infer_result(self, x, weight, eps):
        return (x, x)


class Rope(Operator):
    def __init__(self,):
        super().__init__("Rope")
    
    def infer_result(self, query, key, cos, sin, seqlen):
        return (query, key)


class Inplace(Operator):
    def __init__(self):
        super().__init__("Inplace")
    
    def infer_result(self, input, target, input_index=-1, target_index=-1):
        if target_index == -1:
            return target
        return target[target_index]


class SelfAttentionPAEncoder(Operator):
    def __init__(self):
        super().__init__("SelfAttentionPAEncoder")
    
    def infer_result(self, query, key, value, seqlen, mask, q_head_num, kv_head_num):
        return query


class ReshapeAndCache(Operator):
    def __init__(self):
        super().__init__("ReshapeAndCache")
    
    def infer_result(self, key, value, key_cache, value_cache, kv_indices):
        return key_cache, value_cache


class PagedAttention(Operator):
    def __init__(self):
        super().__init__("PagedAttention")
    
    def infer_result(self, query, key_cache, value_cache, block_table, context_len, mask, q_head_num, kv_head_num, scale):
        return query

class AddRmsNorm(Operator):
    def __init__(self):
        super().__init__("AddRmsNorm")
    
    def infer_result(self, x1, x2, gamma, epsilon):
        return x1, x1, x1
