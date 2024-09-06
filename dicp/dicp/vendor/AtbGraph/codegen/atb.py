import os
import torch
from typing import Any, List
from torch.fx.node import Node

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
from dicp.vendor.AtbGraph.codegen.atb_graph import Graph, parse_graph
from dicp.vendor.AtbGraph.codegen.atb_op import AtbOverrides

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
        # graph.print_readable()

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
            if output not in self.output_tensor_descs['param'].keys():
                import pdb;pdb.set_trace()
                pass
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

        call_body.writeline(f'''host_tensor_dict = {{}}''')
        host_tensors = []
        for tensor in self.atb_graph.hosts:
            node_id = tensor["nodeId"]
            tensor_id = tensor["tensorId"]
            tensor_name = tensor["tensorName"]
            assert tensor_name in self.args
            if tensor_name not in host_tensors:
                call_body.writeline(f'''host_tensor_dict["{tensor_name}"] = {tensor_name}.cpu().tolist()''')
                host_tensors.append(tensor_name)
            call_body.writeline(f'''output_tensor_descs["hostTensors"].append({{"nodeId": {node_id}, "tensorId": {tensor_id}, "value": host_tensor_dict["{tensor_name}"] }})''')

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

    def generate_code(self):
        self.parse_outputs()
        self.atb_graph, self.output_tensor_descs, self.py_output_names = parse_graph(
            self.atb_graph, self.graph_input_names, self.graph_output_names, self.input_args, self.output_args, self.py_output_names)
        return (self.gen_import_code() + self.gen_compile_graph_code() + self.gen_call_func() + self.gen_main_func())
