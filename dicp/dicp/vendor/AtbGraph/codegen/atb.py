import json
import os
import math
import torch
from typing import Any, List
from torch.fx.node import Node
from torch.utils._pytree import tree_map_only
from torch._inductor.utils import IndentedBuffer
from dicp.dynamo_bridge.utils import symint_in_shape, process_sym_name
from dicp.vendor.AscendGraph.codegen.utils import (
    get_ascend_dtype,
    get_cpp_dtype,
    get_ascend_dtype_num,
    get_torch_dtype,
    AclFormat,
    AclDataType,
    get_acl_dtype
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
        
        aten_graph.print_readable()
        graph.print_readable()

        # for modified args return
        self.assign_args = []
        self.cpu_tensor = []
        self.atb_nodes_map = {}
        self.atb_graph_nodes_map = {}
        self.atb_single_nodes_map = {}
        self.atb_single_nodes_list = []
        self.atb_graph_ndoes_list = []
        self.atb_getitem_nodes_list = []
        self.atb_inplace_nodes_list = []
        self.atb_nodes = OrderedDict()
        self.atb_graph = {}
        self.output_nodes = []
        self.sym_input_names = []
        self.atb_getitem_replace_dict = {}
        self.atb_inplace_replace_dict = {}
        self.atb_host_tensor_names = []

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
        if not isinstance(op, list):
            self.atb_nodes[op.op_name] = op
            if isinstance(op, AtbGraphOpearation):
                self.atb_graph_ndoes_list.append(op)
            if isinstance(op, AtbGetItemOperation):
                self.atb_getitem_nodes_list.append(op)
            if isinstance(op, AtbInplaceOperation):
                self.atb_inplace_nodes_list.append(op)
        else:
            import pdb;pdb.set_trace()
            pass

    def get_attr(self, name, target, args, kwargs):
        assert isinstance(target, str)
        attr = self.fetch_attr(target)
        assert (isinstance(attr, torch.Tensor))
        self.args_dict[name] = name
        op = getattr(self.override, 'get_const_attr')(name, attr)
        self.at
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

        output_tensor_descs = {}
        output_tensor_descs_for_create = {}
        for output in self.output_args:
            def get_shape(elem):
                if hasattr(elem, 'meta'):
                    elem = elem.meta['val']
                if isinstance(elem, torch.SymInt) or isinstance(elem, torch.SymBool):
                    return [1], 1
                shape = list(elem.shape)
                if len(shape) == 0:
                    raise RuntimeError("Error handling empty output_shape")
                shape = [process_sym_name(dim) for dim in shape]
                dim_num = len(shape)
                return shape, dim_num
            
            def get_dtype(elem):
                if hasattr(elem, 'meta'):
                    elem = elem.meta['val']
                if isinstance(elem, torch.SymInt):
                    return AclDataType.ACL_INT32.value
                if isinstance(elem, torch.SymBool):
                    return AclDataType.ACL_BOOL.value
                return get_acl_dtype(elem.dtype)
            
            dims, dim_num = get_shape(output)
            dims = f'[{",".join(dims)}]'
            dtype = get_dtype(output)
            info = f''' {{"format": {AclFormat.ACL_FORMAT_ND.value}, "dtype": {dtype}, "dimNum": {dim_num}, "dims": {dims} }} '''
            if output.name in self.atb_getitem_replace_dict.keys():
                output_name = self.atb_getitem_replace_dict[output.name]
            else:
                output_name = output.name
            output_tensor_descs[output_name] = info
            output_tensor_descs_for_create[output_name] = {
                "dtype": str(get_torch_dtype(dtype)),
                "shape": dims
            }
        # gen fixed output shape
        call_body.writeline('''output_tensor_descs = {"outputTensorDescs": [], "hostTensors": []}''')
        graph_input_names = self.atb_graph["inputNames"]
        graph_output_names = self.atb_graph["outputNames"]

        for output in graph_output_names:
            call_body.writeline(f'''output_tensor_descs["outputTensorDescs"].append({output_tensor_descs[output]})''')
            output_name = output if output not in self.atb_inplace_replace_dict.keys() else self.atb_inplace_replace_dict[output]
            if output_name not in self.real_graph_input_names:
                if output in output_tensor_descs_for_create.keys(): 
                    shape = output_tensor_descs_for_create[output]["shape"]
                    dtype = output_tensor_descs_for_create[output]["dtype"]
                else:
                    import pdb;pdb.set_trace()
                    pass
                device = 'npu'
                call_body.writeline(f'''{output} = torch.empty({shape}, dtype={dtype}, device='{device}')''')
            else:
                call_body.writeline(f'''{output} = {output_name}''')
        for tensor in self.atb_graph["hostTensorNames"]:
            node_id = tensor["nodeId"]
            tensor_id = tensor["tensorId"]
            tensor_name = tensor["tensorName"]
            assert tensor_name in self.real_graph_input_names
            call_body.writeline(f'''output_tensor_descs["hostTensors"].append({{"nodeId": {node_id}, "tensorId": {tensor_id}, "value": {tensor_name}.cpu().tolist() }})''')
            # graph_input_names.remove(tensor_name)

        call_body.writeline('''output_tensor_descs_string = json.dumps(output_tensor_descs)''')
        call_body.writeline('''output_shape = output_tensor_descs_string ''')
        call_body.writeline(f'''inputs = [{','.join(graph_input_names)}]''')
        
        call_body.writeline(f'''outputs = [{','.join(graph_output_names)}]''')
        call_body.writeline(f'''import pdb;pdb.set_trace()''')
        call_body.writeline('kernel_cpp_0(inputs, outputs, output_shape)')
        # call_str = ['output_tensor = kernel_cpp_0(inputs, output_shape)']

        # for i, name in enumerate(self.graph_output_names):
        #     if name not in self.symint_outputs:
        #         if name in self.cpu_tensor:
        #             call_str.append(f'{name} = output_tensor[{i}].cpu()')
        #         else:
        #             call_str.append(f'{name} = output_tensor[{i}]')
        #     else:
        #         call_str.extend([f'del {name}',
        #                          f'{name} = int(output_tensor[{i}])'])

        # call_body.writelines(call_str)

        py_output_names = self.preprocess_tensor_names(self.py_output_names)
        del_args = [f'del {x}' for x in self.args if x not in py_output_names]
        call_body.writelines(del_args)
        call_body.writeline("args.clear()")
        call_body.writeline(f"return ({', '.join(py_output_names)})")

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
        return json.dumps(self.atb_graph)

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

    def replace_getitem_name(self, name_list):
        for i in range(len(name_list)):
            if name_list[i] in self.atb_getitem_replace_dict.keys():
                name_list[i] = self.atb_getitem_replace_dict[name_list[i]]
        return name_list
    
    def replace_inplace_name(self, name_list):
        for i in range(len(name_list)):
            if name_list[i] in self.atb_inplace_replace_dict.keys():
                name_list[i] = self.atb_inplace_replace_dict[name_list[i]]
        return name_list
    
    def preprocess_tensor_names(self, name_list):
        name_list = self.replace_getitem_name(name_list)
        # name_list = self.replace_inplace_name(name_list)
        return name_list

    def process_atb_graph(self):
        # process all host tensor
        for k, v in self.atb_nodes.items():
            if isinstance(v, AtbSingleOperator):
                if v.has_host_inputs:
                    self.atb_host_tensor_names.extend(v.host_input_names)

        # process all getitem operation
        for getitem_node in self.atb_getitem_nodes_list:
            input_name = getitem_node.input_name
            index = getitem_node.index
            op_name = getitem_node.op_name
            self.atb_getitem_replace_dict[op_name] = f"{input_name}_{index}"
            del self.atb_nodes[op_name]
        
        # process all inplace operation
        for inplace_node in self.atb_inplace_nodes_list:
            input_name = inplace_node.input_name
            target_name = inplace_node.target_name
            op_name = inplace_node.op_name
            self.atb_inplace_replace_dict[input_name] = target_name
            del self.atb_nodes[op_name]
 
        # process all graph operations
        for graph_node in self.atb_graph_ndoes_list:
            input_names = []
            output_names = []
            graph_node.node_size = len(graph_node.node_names)
            graph_single_ops = {}
            for single_op_name in graph_node.node_names:
                single_op = self.atb_nodes[single_op_name]
                graph_single_ops[single_op_name] = single_op
                del self.atb_nodes[single_op.op_name]
                input_names.extend(self.preprocess_tensor_names(single_op.input_names))
                output_names.extend(self.preprocess_tensor_names(single_op.output_names))
                graph_node.nodes.append(single_op.build())
            # internal_names = output_names - graph_node.output_names
            graph_node.output_names = self.preprocess_tensor_names(graph_node.output_names)
            graph_node.internal_names = [x for x in output_names if x not in graph_node.output_names]
            graph_node.input_names = [x for x in input_names if x not in graph_node.internal_names]

            self.atb_nodes[graph_node.op_name] = graph_node.build()
        
        for k, v in self.atb_nodes.items():
            if not isinstance(v, dict):
                self.atb_nodes[k] = v.build()
            print(f'k: {k}  value: {self.atb_nodes[k]}')

        # generate inputs/outputs/internals
        self.real_graph_input_names = []
        for input in self.graph_input_names:
            if input in self.sym_input_names:
                continue
            self.real_graph_input_names.append(input)
        input_output_names = []
        self.atb_graph["name"] = str(self.graph_id)
        self.atb_graph["outputNames"] = self.preprocess_tensor_names(self.graph_output_names)
        self.atb_graph["inputNames"] = self.preprocess_tensor_names(self.real_graph_input_names)
        self.atb_graph["nodes"] = []
        for k, v in self.atb_nodes.items():
            input_names = self.preprocess_tensor_names(v["value"]["inputNames"])
            output_names = self.preprocess_tensor_names(v["value"]["outputNames"])
            for input in input_names:
                if input not in input_output_names:
                    input_output_names.append(input)
            for output in output_names:
                if output not in input_output_names:
                    input_output_names.append(output)
            self.atb_graph["nodes"].append(v)
        internal_names = []
        for tensor in input_output_names:
            if tensor not in self.atb_graph["inputNames"] and tensor not in self.atb_graph["outputNames"]:
                internal_names.append(tensor)
        self.atb_graph["internalNames"] = internal_names
        
        graph_host_names = []
        for ni, node in enumerate(self.atb_graph["nodes"]):
            input_names = node["value"]["inputNames"]
            for ti, name in enumerate(input_names):
                if name in self.atb_host_tensor_names:
                    graph_host_names.append({"nodeId": ni, "tensorId": ti, "tensorName": name})
        self.atb_graph["hostTensorNames"] = graph_host_names

    def generate_code(self):
        self.parse_outputs()
        self.process_atb_graph()
        return (self.gen_import_code() + self.gen_compile_graph_code() + self.gen_call_func() + self.gen_main_func())


class AtbSingleOperator:
    def __init__(self, op_name: str, op_type: str):
        self.op_name = op_name
        self.op_type = op_type
        self.param = {}
        self.input_names = []
        self.output_names = []
        self.has_host_inputs = False
        self.host_input_names = []
    
    def set_input(self, x):
        self.input_names = x 
    
    def set_output(self, x):
        self.output_names = x
        
    def add_input(self, x):
        self.input_names.append(x)
    
    def add_output(self, x):
        self.output_names.append(x)
    
    def set_param(self, x):
        if not isinstance(x, dict):
            x = infer_param.to_dict(x)
        self.param = x

    def build(self):
        node = {
            "nodeType": "singleOperation",
            "value": {
                "name": self.op_name,
                "type": self.op_type,
                "param": self.param,
                "inputNames": self.input_names,
                "outputNames": self.output_names,
                "hasHostInputs": self.has_host_inputs,
                "hostInputNames": self.host_input_names,
            },
        }
        return node


class AtbGetItemOperation:
    def __init__(self, name: str):
        self.op_name = name
        self.op_type = "getitemOperation"
        self.input_name = ""
        self.index = -1


class AtbInplaceOperation:
    def __init__(self, name: str):
        self.op_name = name
        self.op_type = "inplaceOperation"
        self.input_name = ""
        self.target_name = ""


class AtbGraphOpearation:
    def __init__(self, name: str):
        self.op_name = name
        self.op_type = "graphOperation"
        self.nodes = []
        self.input_names = []
        self.output_names = []
        self.internal_names = []
        self.node_size = -1
        self.node_names = []
        self.has_host_inputs = False
        self.host_input_names = []
    
    def set_node_names(self, x):
        self.node_names = x
    
    def add_node_name(self, x):
        self.node_names.append(x)
    
    def set_input(self, x):
        self.input_names = x
    
    def set_output(self, x):
        self.output_names = x
    
    def set_internal(self, x):
        self.internal_names = x
    
    def set_nodes(self, x):
        self.nodes = x

    def add_input(self, x):
        self.input_names.append(x)
    
    def add_output(self, x):
        self.output_names.append(x)
    
    def add_internal(self, x):
        self.internal_names.append(x)
    
    def add_node(self, x):
        self.nodes.append(x)

    def build(self):
        graph = {
            "nodeType": "graphOperation",
            "value": {
            "nodes": self.nodes,
            "inputNames": self.input_names,
            "outputNames": self.output_names,
            "internalNames": self.internal_names,
            "nodeSize": self.node_size,
            "hasHostInputs": self.has_host_inputs,
            "hostInputNames": self.host_input_names,
            }
        }
        return graph

class AtbOverrides:
    @staticmethod
    def gen_args(op_var, args_dict, args):
        src_code = IndentedBuffer()
        args_str = [op_var]
        args_str.extend(tree_map_only(Node, lambda x: args_dict[x.name], args))
        return src_code, args_str

    @staticmethod
    def Linear(name, a, b, bias, trans_a, trans_b, out_dtype=None):
        op = AtbSingleOperator(name, "LinearOperation")
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
        op = AtbSingleOperator(name, "ElewiseOperation")
        param = infer_param.ElewiseParam()
        param.elewiseType = infer_param.ElewiseType.ELEWISE_ADD

        op.set_input([x, y])
        op.set_param(param)
        op.set_output([name])
        return op

    def Graph(name, *args, **kwargs):
        outputs = kwargs['output']
        
        if not isinstance(outputs, list):
            outputs = [outputs]
        graph_output_names = [str(x) for x in outputs]

        op = AtbGraphOpearation(name)
        op.set_node_names(list(args))
        op.set_output(graph_output_names)
        return op

    def GetItem(name, x, index):
        op = AtbGetItemOperation(name)
        op.input_name = x
        op.index = index
        return op

    def RmsNorm(name, x, w, eps):
        op = AtbSingleOperator(name, "RmsNormOperation")
        param = infer_param.RmsNormParam()
        param.layerType = infer_param.RmsNormType.RMS_NORM_NORM
        param.normParam.epsilon = eps
        
        op.set_input([x, w])
        op.set_param(param)
        op.set_output([f"{name}_0"])
        return op
    
    def Rope(name, query, key, cos, sin, seqlen):
        op = AtbSingleOperator(name, "RopeOperation") 
        param = infer_param.RopeParam()
        param.rotaryCoeff = 2
        
        op.set_input([query, key, cos, sin, seqlen])
        op.set_param(param)
        op.set_output([f"{name}_0", f"{name}_1"])
        return op

    def Inplace(name, input, target, input_index=-1, target_index=-1):
        op = AtbInplaceOperation(name)
    
        op.input_name = input if input_index == -1 else f"{input}_{input_index}"
        op.target_name = target if target_index == -1 else f"{target}_{target_index}"
        return op

    def SelfAttentionPAEncoder(name, query, key, value, seqlen, mask, q_head_num, kv_head_num):
        op = AtbSingleOperator(name, "SelfAttentionOperation")
        param = infer_param.SelfAttentionParam()
        param.calcType = infer_param.SelfAttentionCalcType.PA_ENCODER
        param.kernelType = infer_param.SelfAttentionKernelType.KERNELTYPE_DEFAULT
        param.clampType = infer_param.SelfAttentionClampType.CLAMP_TYPE_UNDEFINED
        param.headNum = q_head_num
        param.kvHeadNum = kv_head_num

        if mask is not None:
            param.maskType = infer_param.SelfAttentionMaskType.MASK_TYPE_NORM
            op.set_input([query, key, value, mask, seqlen])
        else:
            param.maskType = infer_param.SelfAttentionMaskType.MASK_TYPE_UNDEFINED
            op.set_input([query, key, value, seqlen])
            
        op.set_param(param)
        op.set_output([name])
        op.has_host_inputs = True
        op.host_input_names.append(seqlen)
        return op

    def ReshapeAndCache(name, key, value, key_cache, value_cache, kv_indices):
        op = AtbSingleOperator(name, "ReshapeAndCacheOperation")
        param = infer_param.ReshapeAndCacheParam()
        
        op.set_param(param)
        op.set_input([key, value, key_cache, value_cache, kv_indices])
        op.set_output([f"{name}_0", f"{name}_1"])
        return op

    def PagedAttention(name, query, key_cache, value_cache, block_table, context_len, mask, q_head_num, kv_head_num, scale):
        op = AtbSingleOperator(name, "PagedAttentionOperation")
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
        op.host_input_names.append(context_len)
        return op

    def AddRmsNorm(name, x1, x2, gamma, epsilon):
        op = AtbSingleOperator(name, "AddRmsNormOperation")
        param = infer_param.AddRmsNormParam()
        param.epsilon = epsilon
        op.set_param(param)
        op.set_input([x1, x2, gamma])
        op.set_output([f"{name}_0", f"{name}_1", f"{name}_2"])
        return op
