import json
import os
import math
import torch
from typing import Any, List
from torch.fx.node import Node
from torch.utils._pytree import tree_map_only
from torch._inductor.utils import IndentedBuffer
from dicp.dynamo_bridge.utils import symint_in_shape, process_sym_name
from dicp.vendor.AtbGraph.codegen.utils import (
    get_ascend_dtype,
    get_cpp_dtype,
    get_ascend_dtype_num,
    get_torch_dtype,
    AclFormat,
    AclDataType,
    get_acl_dtype,
    remove_duplicates,
)
from dicp.vendor.AtbGraph.codegen.atb_graph import (Operation,
                                                    GetItemOperation,
                                                    InplaceOperation,
                                                    AtbTupleOperation,
                                                    ViewOperation,
                                                    GraphOpearation,
                                                    Graph,
                                                    parse_graph,
                                                    )
from collections import OrderedDict
import dicp.vendor.AtbGraph.codegen.atb_infer_param as infer_param

graph_id = 0

precision_check = bool(os.environ.get("DICP_ASCEND_PRECISION_CHECK", False))


def get_graph_id():
    global graph_id
    graph_id = graph_id + 1
    return graph_id


def process_name(name, target):
    if hasattr(target, "name"):
        real_op = target.name().split('::')[-1]
        if real_op.find('.') != -1:
            real_op = real_op.split('.')[0]
    else:
        real_op = name.rsplit('_', 1)[0] if name[-1].isdigit() else name
    return real_op


class AtbCodegen(torch.fx.Interpreter):
    def __init__(self, graph, aten_graph=None, folder=None, graph_key=None):
        self.graph = graph
        self.aten_graph = aten_graph
        self.override = AtbOverrides

        self.import_code = IndentedBuffer()
        self.build_graph_code = IndentedBuffer(initial_indent=1)

        self.graph_id = str(get_graph_id())
        self.args_dict = {}
        self.input_args = []
        self.output_args = []

        self.dynamic_inputs = []
        self.dynamic_shape = []
        self.actual_shape = []
        self.dynamic_index = []
        self.symint_outputs = []
        self.sym_input_names = []

        self.data_nodes = []
        self.common_nodes = []
        self.graph_input_names = []
        self.py_output_names = []
        self.graph_output_names = []
        self.build_options = []

        self.folder = folder
        self.graph_key = graph_key

        self.sym_to_inputs = {}
        self.sym_in_args = {}
        
        # aten_graph.print_readable()
        graph.print_readable()

        # for modified args return
        self.assign_args = []
        self.cpu_tensor = []
        self.atb_graph = Graph(str(get_graph_id()))

        super().__init__(graph)

    def placeholder(self, name, target, args, kwargs):
        self.args_dict[name] = name
        self.input_args.append(self.cur_node)

        fake_tensor = self.cur_node.meta['val']
        format = "ND"
        index = -1

        if isinstance(fake_tensor, torch.SymInt):
            dims = [1]
            data_type = "INT32"
            format = "ND"
            self.sym_to_inputs[fake_tensor.node.str()] = name
            self.sym_input_names.append(name)
        elif symint_in_shape(fake_tensor.shape):
            # mention symint position in args
            # dynamic shape feature
            for idx, dim in enumerate(fake_tensor.shape):
                if isinstance(dim, torch.SymInt):
                    st = dim.node.str()
                    if st not in self.sym_in_args:
                        self.sym_in_args[st] = (name, idx)

            # deal with dynamic shape -1
            shape = [-1 if isinstance(elem, torch.SymInt)
                     else elem for elem in fake_tensor.shape]
            actual_shape = [elem.node.str() if isinstance(
                elem, torch.SymInt) else str(elem) for elem in fake_tensor.shape]
            self.dynamic_inputs.append(self.args_dict[name])
            self.dynamic_shape.append(shape)
            self.actual_shape.append(actual_shape)
            self.dynamic_index.append(len(self.graph_input_names))
            dims = shape
            data_type = get_ascend_dtype(fake_tensor.dtype).upper()
        else:
            dims = list(fake_tensor.shape)
            data_type = get_ascend_dtype(fake_tensor.dtype).upper()

        if 'native_memory_format' in self.cur_node.meta:
            format = self.cur_node.meta['native_memory_format']
        # gen data_nodes
        self.data_nodes.append({
            "op_name": self.args_dict[name],
            "op_type": "Data",
            "dims": dims,
            "format": format,
            "data_type": data_type,
            "cpp_data_type": data_type,
            "index": index
        })
        self.graph_input_names.append(self.args_dict[name])

    def call_function(self, name, target, args, kwargs):
        if name not in self.args_dict.keys():
            self.args_dict[name] = name

        if hasattr(self.cur_node, 'meta'):
            if 'prop' in self.cur_node.meta and 'cpu_tensor' in self.cur_node.meta['prop']:
                self.cpu_tensor.append(self.cur_node.meta['prop']['cpu_tensor'])
            if 'prop' in self.cur_node.meta and 'assign_args' in self.cur_node.meta['prop']:
                self.assign_args.append(self.cur_node.meta['prop']['assign_args'])

        _, args_list = AtbOverrides.gen_args(
            self.args_dict[name], self.args_dict, args)
        real_op = process_name(name, target)
        op = getattr(self.override, real_op)(*args_list, **kwargs)
        self.atb_graph.add_node(op)

    def get_attr(self, name, target, args, kwargs):
        assert isinstance(target, str)
        attr = self.fetch_attr(target)
        assert (isinstance(attr, torch.Tensor))
        self.args_dict[name] = name
        op = getattr(self.override, 'get_const_attr')(name, attr)
        self.common_nodes.append(op)

    def call_method(self, name, target, args, kwargs):
        pass

    def output(self, name, target, args, kwargs):
        for arg in args:
            self.output_args.extend(arg)

    def run_node(self, n: Node) -> Any:
        self.cur_node = n
        op = n.op
        name = n.name
        target = n.target
        args = n.args
        kwargs = n.kwargs

        assert isinstance(args, tuple)
        assert isinstance(kwargs, dict)

        return getattr(self, op)(name, target, args, kwargs)

    def codegen(self):
        self.run()
        return self.generate_code()

    def parse_outputs(self):
        symint_inputs = self.sym_to_inputs.values()
        real_output_args = []
        for node in self.output_args:
            if isinstance(node, torch.fx.node.Node):
                name = self.args_dict[node.name]
                self.py_output_names.append(name)
                if name in self.graph_output_names or name in self.graph_input_names:
                    continue
                else:
                    real_output_args.append(node)
                    self.graph_output_names.append(name)
                if name in symint_inputs:
                    self.symint_outputs.append(name)
            else:
                self.py_output_names.append(str(node))
        self.output_args = real_output_args

    def gen_import_code(self):
        self.import_code.splice(
            """
                import torch
                import torch_npu
                import random
                import json
                from torch import empty_strided, as_strided, device
                from dicp.dynamo_bridge.compile import AsyncCompileKernel
                from dicp.vendor.AtbGraph.compile_job import AtbCompileJob

                aten = torch.ops.aten
                assert_size_stride = torch._C._dynamo.guards.assert_size_stride

                def check_tensor(a, b, atol=5e-2, rtol=1e-2):
                    if not torch.allclose(a, b, atol=atol, rtol=rtol, equal_nan=True):
                        import pdb;pdb.set_trace()
                        pass
            """, strip=True
        )
        return self.import_code.getvalue()

    def operator_in_str(self, st):
        for op in ['+', '-', '*', '/']:
            if op in st:
                return True
        return False

    def gen_call_func(self):
        # TODO check scalar input
        call_body = IndentedBuffer()
        self.args = [self.args_dict[x.name] for x in self.input_args]
        if len(self.args) == 1:
            call_body.writeline(f"{self.args[0]} = args[0]")
        else:
            call_body.writeline(f"({','.join(self.args)}) = args")

        # assign SymInt to InputArgs relationship
        if len(self.sym_in_args) > 0:
            for key in self.sym_in_args.keys():
                if not key.isdigit() and not self.operator_in_str(key):
                    call_body.writeline(f"{key} = {self.sym_in_args[key][0]}.shape[{self.sym_in_args[key][1]}]")
        if len(self.sym_to_inputs) > 0:
            for key in self.sym_to_inputs.keys():
                if not key.isdigit() and not self.operator_in_str(key):
                    call_body.writeline(f"{key} = {self.sym_to_inputs[key]}")
            
        # gen fixed output shape
        call_body.writeline('''output_tensor_descs = {"outputTensorDescs": [], "hostTensors": []}''')
        graph_input_names = self.atb_graph.inputs
        graph_output_names = self.atb_graph.outputs

        for output in graph_output_names:
            param = self.output_tensor_descs['param'][output]
            create_info = self.output_tensor_descs['create'][output]
            call_body.writeline(f'''output_tensor_descs["outputTensorDescs"].append({param})''')
            if create_info['input'] is None:
                device = 'npu'
                dtype = create_info['dtype']
                shape = create_info['shape']
                call_body.writeline(f'''{output} = torch.empty({shape}, dtype={dtype}, device='{device}')''')
            elif create_info['need_reshape']:
                shape = create_info['shape']
                input = create_info['input']
                call_body.writeline(f'''{output} = {input}.view({shape})''')
            else:
                input = create_info['input']
                call_body.writeline(f'''{output} = {input}''')

        for tensor in self.atb_graph.hosts:
            node_id = tensor["nodeId"]
            tensor_id = tensor["tensorId"]
            tensor_name = tensor["tensorName"]
            assert tensor_name in self.args
            call_body.writeline(f'''output_tensor_descs["hostTensors"].append({{"nodeId": {node_id}, "tensorId": {tensor_id}, "value": {tensor_name}.cpu().tolist() }})''')

        call_body.writeline('''output_tensor_descs_string = json.dumps(output_tensor_descs)''')
        call_body.writeline('''output_shape = output_tensor_descs_string ''')
        call_body.writeline(f'''inputs = [{','.join(graph_input_names)}]''')
        
        call_body.writeline(f'''outputs = [{','.join(graph_output_names)}]''')
        # call_body.writeline(f'''import pdb;pdb.set_trace()''')
        call_body.writeline('kernel_cpp_0(inputs, outputs, output_shape)')

        # py_output_names = self.preprocess_tensor_names(self.py_output_names)
        # del_args = [f'del {x}' for x in self.args if x not in py_output_names]
        # call_body.writelines(del_args)
        # call_body.writeline("args.clear()")
        call_body.writeline(f"return ({', '.join(self.py_output_names)})")

        call_func = IndentedBuffer()
        call_func.writeline("def call(args):")
        with call_func.indent():
            call_func.splice(call_body)

        return call_func.getvalue()

    def gen_main_func(self):
        main_body = IndentedBuffer()
        main_body.splice(
            """
                from torch._dynamo.testing import rand_strided
                from torch._inductor.utils import print_performance
            """, strip=True
        )

        py_rand_inputs = []
        for i in range(len(self.input_args)):
            node = self.input_args[i]
            name = self.args[i]
            val = node.meta['val']
            if isinstance(val, torch.SymInt):
                code_str = f'''{name} = random.randint(0, 4)'''
            else:
                shape = str(tuple(val.size()))
                stride = str(tuple(val.stride()))
                device = val.device.type
                dtype = str(val.dtype)
                code_str = f'''{name} = rand_strided({shape}, {stride}, device='{device}', dtype={dtype})'''
            py_rand_inputs.append(code_str)
        main_body.writelines(py_rand_inputs)
        main_body.writeline(
            f"print_performance(lambda: call([{', '.join(self.args)}]))")

        main_func = IndentedBuffer()
        main_func.writeline("""if __name__ == "__main__":""")
        with main_func.indent():
            main_func.splice(main_body)
        return main_func.getvalue()


    def expand_symint(self, d, k):
        if isinstance(d[k], torch.SymInt):
            if d[k].node.str().isdigit():
                d[k] = d[k].node.hint
            else:
                raise RuntimeError("expand_symint failed!")

    def remove_symint(self, cur):
        if isinstance(cur, list):
            for idx in range(len(cur)):
                self.expand_symint(cur, idx)
                self.remove_symint(cur[idx])
        elif isinstance(cur, dict):
            for k in cur.keys():
                self.expand_symint(cur, k)
                self.remove_symint(cur[k])

    def gen_graph_json(self):        
        return self.atb_graph.to_json()

    def gen_compile_graph_code(self):
        compile_graph_code = IndentedBuffer()
        graph_json = self.gen_graph_json()
        compile_graph_code.splice(
            f"""
                atb_compile_job = AtbCompileJob('''{graph_json}''')
                async_compile = AsyncCompileKernel()
                kernel_cpp_0 = async_compile.compile_kernel(atb_compile_job)
            """, strip=True
        )
        compile_graph_code.writeline('async_compile.wait(globals())')
        compile_graph_code.writeline('del async_compile')
        return compile_graph_code.getvalue()

    def process_atb_graph(self):
        def process_tuple_operations():
            for tuple_op in self.atb_tuple_nodes_list:
                self.atb_tuple_replace_dict[tuple_op.op_name] = tuple_op
                
        def process_view_operations():
            for view_node in self.atb_view_nodes_list:
                while view_node.input_name in self.atb_view_tensor_dict.keys():
                    input_name = self.atb_view_tensor_dict[view_node.input_name].input_name
                    view_node.input_name = input_name
                self.atb_view_tensor_dict[view_node.op_name] = view_node
                del self.atb_nodes[view_node.op_name]
                
        def post_delete_view_operations():
            for k, v in self.atb_nodes.items():
                if not isinstance(v, AtbViewOperation):
                    continue
                del self.atb_nodes[k]
        
        def extend_host_tensor_names():
            for k, v in self.atb_nodes.items():
                if isinstance(v, AtbSingleOperator) and v.has_host_inputs:
                    self.atb_host_tensor_names.extend(v.host_input_names)

        def process_getitem_operations():
            for getitem_node in self.atb_getitem_nodes_list:
                self.atb_getitem_replace_dict[
                    getitem_node.op_name] = f"{getitem_node.input_name}_{getitem_node.index}"
                del self.atb_nodes[getitem_node.op_name]

        def process_inplace_operations():
            for inplace_node in self.atb_inplace_nodes_list:
                self.atb_inplace_replace_dict[inplace_node.input_name] = inplace_node.target_name
                del self.atb_nodes[inplace_node.op_name]

        def process_graph_operations():
            for graph_node in self.atb_graph_ndoes_list:
                input_names, output_names = [], []    
                graph_single_ops = {}
                for single_op_name in graph_node.node_names:
                    if single_op_name in self.atb_view_tensor_dict.keys():
                        input_names.append(self.atb_view_tensor_dict[single_op_name].op_name)
                        continue
                    if single_op_name in self.atb_getitem_replace_dict.keys():
                        input_names.append(self.atb_getitem_replace_dict[single_op_name])
                        continue
                    if single_op_name not in self.atb_nodes.keys():
                        continue
                    
                    single_op = self.atb_nodes[single_op_name]
                    del self.atb_nodes[single_op.op_name]
                    graph_single_ops[single_op_name] = single_op
                    

                    # need reshape inputs
                    if any(name in self.atb_view_tensor_dict.keys() for name in single_op.input_names):
                        single_op.has_reshape_inputs = True
                        single_op.reshape_inputs = []
                        for i, t in enumerate(single_op.input_names):
                            if t in self.atb_view_tensor_dict.keys():
                                single_op.reshape_inputs.append(self.atb_view_tensor_dict[t].target_reshape_info)
                                single_op.input_names[i] = self.atb_view_tensor_dict[t].input_name
                            else:
                                single_op.reshape_inputs.append({"reshapeType": "None"})

                    input_names.extend(
                        self.preprocess_tensor_names(single_op.input_names))
                    input_names = remove_duplicates(input_names)
                    output_names.extend(
                        self.replace_inplace_name(self.preprocess_tensor_names(single_op.output_names)))
                    graph_node.nodes.append(single_op.build())
                # internal_names = output_names - graph_node.output_names
                graph_node.output_names = self.preprocess_tensor_names(
                    graph_node.output_names)
                graph_node.internal_names = [
                    x for x in output_names if x not in graph_node.output_names]
                graph_node.input_names = [
                    x for x in input_names if x not in graph_node.internal_names]
                graph_node.node_size = len(graph_node.nodes)
                self.atb_nodes[graph_node.op_name] = graph_node.build()

        def build_remaining_nodes():
            for k, v in self.atb_nodes.items():
                if not isinstance(v, dict):
                    if any(name in self.atb_view_tensor_dict.keys() for name in v.input_names):
                        v.has_reshape_inputs = True
                        v.reshape_inputs = []
                        for i, t  in enumerate(v.input_names):
                            if t in self.atb_view_tensor_dict.keys():
                                v.reshape_inputs.append(self.atb_view_tensor_dict[t].target_reshape_info)
                                v.input_names[i] = self.atb_view_tensor_dict[t].input_name
                            else:
                                v.reshape_inputs.append({"reshapeType": "None"})
                    self.atb_nodes[k] = v.build()
                # print(f'k: {k}  value: {self.atb_nodes[k]}')

        def generate_inputs_outputs_internals():
            self.real_graph_input_names = [
                input for input in self.graph_input_names if input not in self.sym_input_names]
            input_output_names = []
            self.atb_graph["name"] = str(self.graph_id)
            self.atb_graph["outputNames"] = self.preprocess_tensor_names(
                self.graph_output_names)
            self.atb_graph["inputNames"] = self.preprocess_tensor_names(
                self.real_graph_input_names)
            self.atb_graph["nodes"] = []
            for k, v in self.atb_nodes.items():
                input_names = self.preprocess_tensor_names(
                    v["value"]["inputNames"])
                output_names = self.preprocess_tensor_names(
                    v["value"]["outputNames"])
                for name in input_names + output_names:
                    if name not in input_output_names:
                        input_output_names.append(name)
                self.atb_graph["nodes"].append(v)
            self.atb_graph['internalNames'] = []
            for tensor_name in input_output_names:
                if tensor_name in self.atb_graph["inputNames"] or tensor_name in self.atb_graph["outputNames"]:
                    continue
                if tensor_name in self.atb_inplace_replace_dict.keys():
                    self.atb_graph['outputNames'].append(tensor_name)
                    continue
                self.atb_graph['internalNames'].append(tensor_name)
            # self.atb_graph["internalNames"] = [
            #     tensor for tensor in input_output_names if tensor not in self.atb_graph["inputNames"] and tensor not in self.atb_graph["outputNames"]]

        def generate_host_tensor_names():
            graph_host_names = []
            for ni, node in enumerate(self.atb_graph["nodes"]):
                for ti, name in enumerate(node["value"]["inputNames"]):
                    if name in self.atb_host_tensor_names:
                        graph_host_names.append(
                            {"nodeId": ni, "tensorId": ti, "tensorName": name})
            self.atb_graph["hostTensorNames"] = graph_host_names

        extend_host_tensor_names()
        process_tuple_operations()
        process_getitem_operations()
        process_inplace_operations()
        process_view_operations()
        process_graph_operations()
        build_remaining_nodes()
        generate_inputs_outputs_internals()
        generate_host_tensor_names()
        

    def generate_code(self):
        self.parse_outputs()
        self.atb_graph, self.output_tensor_descs, self.py_output_names = parse_graph(
            self.atb_graph, self.graph_input_names, self.graph_output_names, self.input_args, self.output_args, self.py_output_names)
        return (self.gen_import_code() + self.gen_compile_graph_code() + self.gen_call_func() + self.gen_main_func())

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
