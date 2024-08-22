import math
import torch
from torch.fx.node import Node
from torch.utils._pytree import tree_map_only
from torch._inductor.utils import IndentedBuffer
import dicp.vendor.AtbGraph.codegen.atb_infer_param as infer_param
from dicp.vendor.AtbGraph.codegen.atb_graph import Operation, GetItemOperation, GraphOpearation, InplaceOperation, ViewOperation

class AtbOverrides:
    @staticmethod
    def gen_args(op_var, args_dict, args):
        src_code = IndentedBuffer()
        args_str = [op_var]
        args_str.extend(tree_map_only(Node, lambda x: args_dict[x.name], args))
        return src_code, args_str

    @staticmethod
    def Linear(name, a, b, bias, trans_a, trans_b, out_dtype=None):
        op = Operation(name, "LinearOperation")
        param = infer_param.LinearParam()
        param.transposeA = trans_a
        param.transposeB = trans_b

        op.set_input([a, b])
        if bias:
            param.hasBias = True
            op.add_input(bias)
        else:
            param.hasBias = False
        if out_dtype:
            assert "now out_dtype cannot set!"
        op.set_param(param)
        op.set_output([name])
        return op

    @staticmethod
    def Add(name, x, y):
        op = Operation(name, "ElewiseOperation")
        param = infer_param.ElewiseParam()
        param.elewiseType = infer_param.ElewiseType.ELEWISE_ADD

        op.set_input([x, y])
        op.set_param(param)
        op.set_output([name])
        return op

    def Mul(name, x, y):
        op = Operation(name, "ElewiseOperation")
        param = infer_param.ElewiseParam()
        param.elewiseType = infer_param.ElewiseType.ELEWISE_MUL

        op.set_input([x, y])
        op.set_param(param)
        op.set_output([name])
        return op

    def Graph(name, *args, **kwargs):
        outputs = kwargs['output']
        if not isinstance(outputs, list):
            outputs = [outputs]

        graph_output_names = []
        for x in outputs:
            if isinstance(x, torch.fx.node.Node) and isinstance(x.meta['val'], list):
                meta_val = x.meta['val']
                if len(meta_val) != 1:
                    node_name = str(x)
                    for i in range(len(meta_val)):
                        graph_output_names.append(f"{node_name}_{i}")
                    continue
            graph_output_names.append(str(x))

        op = GraphOpearation(name)
        op.set_node_names(list(args))
        op.set_output(graph_output_names)
        return op

    def GetItem(name, x, index):
        op = GetItemOperation(name)
        op.set_input([x])
        op.index = index
        op.set_output([name])
        return op

    def RmsNorm(name, x, w, eps):
        op = Operation(name, "RmsNormOperation")
        param = infer_param.RmsNormParam()
        param.layerType = infer_param.RmsNormType.RMS_NORM_NORM
        param.normParam.epsilon = eps

        op.set_input([x, w])
        op.set_param(param)
        op.set_output([name])
        return op

    def Rope(name, query, key, cos, sin, seqlen):
        op = Operation(name, "RopeOperation")
        param = infer_param.RopeParam()
        param.rotaryCoeff = 2

        op.set_input([query, key, cos, sin, seqlen])
        op.set_param(param)
        op.set_output([f"{name}_0", f"{name}_1"])
        return op

    def Inplace(name, input, target, input_index=-1, target_index=-1):
        op = InplaceOperation(name)
        op.input_index = input_index
        op.target_index = target_index
        op.target = target
        op.set_input([input])
        op.set_output([name])
        return op

    def SelfAttentionPAEncoder(name, query, key, value, seqlen, mask, q_head_num, kv_head_num):
        op = Operation(name, "SelfAttentionOperation")
        param = infer_param.SelfAttentionParam()
        param.calcType = infer_param.SelfAttentionCalcType.PA_ENCODER
        param.kernelType = infer_param.SelfAttentionKernelType.KERNELTYPE_DEFAULT
        # param.kernelType = infer_param.SelfAttentionKernelType.KERNELTYPE_HIGH_PRECISION
        param.clampType = infer_param.SelfAttentionClampType.CLAMP_TYPE_UNDEFINED
        param.headNum = q_head_num
        param.kvHeadNum = kv_head_num
        param.qkScale = 1. / math.sqrt(128)
        param.isTriuMask = 1

        if mask is not None:
            param.maskType = infer_param.SelfAttentionMaskType.MASK_TYPE_NORM
            op.set_input([query, key, value, mask, seqlen])
        else:
            param.maskType = infer_param.SelfAttentionMaskType.MASK_TYPE_UNDEFINED
            op.set_input([query, key, value, seqlen])

        op.set_param(param)
        op.set_output([name])
        op.has_host_inputs = True
        op.host_inputs.append(seqlen)
        return op

    def ReshapeAndCache(name, key, value, key_cache, value_cache, kv_indices):
        op = Operation(name, "ReshapeAndCacheOperation")
        param = infer_param.ReshapeAndCacheParam()

        op.set_param(param)
        op.set_input([key, value, key_cache, value_cache, kv_indices])
        op.set_output([f"{name}_0", f"{name}_1"])
        return op

    def PagedAttention(name, query, key_cache, value_cache, block_table, context_len, mask, q_head_num, kv_head_num, scale):
        op = Operation(name, "PagedAttentionOperation")
        param = infer_param.PagedAttentionParam()
        param.headNum = q_head_num
        param.kvHeadNum = kv_head_num
        param.qkScale = scale

        if mask is not None:
            param.maskType = infer_param.PagedAttentionMaskType.MASK_TYPE_NORM
            op.set_input([query, key_cache, value_cache, block_table, context_len, mask])
        else:
            param.maskType = infer_param.PagedAttentionMaskType.UNDEFINED
            op.set_input([query, key_cache, value_cache, block_table, context_len])
        op.set_param(param)
        op.set_output([name])
        op.has_host_inputs = True
        op.host_inputs.append(context_len)
        return op

    def AddRmsNorm(name, x1, x2, gamma, epsilon):
        op = Operation(name, "AddRmsNormOperation")
        param = infer_param.AddRmsNormParam()
        param.epsilon = epsilon
        op.set_param(param)
        op.set_input([x1, x2, gamma])
        op.set_output([f"{name}_0", f"{name}_1", f"{name}_2"])
        return op

    def Transpose(name, x, perm):
        op = Operation(name, "TransposeOperation")
        param = infer_param.TransposeParam(perm)
        # param.perm = perm
        op.set_param(param)
        op.set_input([x])
        op.set_output([name])
        return op

    def View(name, input, size):
        op = ViewOperation(name)
        op.add_input(input)
        op.add_output(name)
        op.target_shape = size
        op.target_reshape_info = {
            "reshapeType": "view",
            "dimNum": len(size),
            "dims": size,
        }
        return op

    def SplitSharing(name, x, size, dim):
        op = Operation(name, "SplitOperation") 
        param = infer_param.SplitParam()
        param.splitDim = dim
        param.splitNum = len(size)
        op.set_param(param)
        op.set_input([x])
        if len(size) == 2:
            op.set_output([f"{name}_0", f"{name}_1"])
        else:
            op.set_output([f"{name}_0", f"{name}_1", f"{name}_2"])
        return op

    def Swish(name, x, scale=1.0, dim=-1):
        op = Operation(name, "ActivationOperation")
        param = infer_param.ActivationParam()
        param.activationType = infer_param.ActivationType.ACTIVATION_SWISH.value
        param.scale = scale
        param.dim = dim
        op.set_param(param)
        op.set_input([x])
        op.set_output([name])
        return op
