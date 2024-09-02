import copy
import torch
import json
from collections import OrderedDict

import dicp.vendor.AtbGraph.codegen.atb_infer_param as infer_param
from dicp.dynamo_bridge.utils import process_sym_name
from dicp.vendor.AtbGraph.codegen.utils import AclDataType, AclFormat, get_acl_dtype, get_torch_dtype

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

class Operation:
    def __init__(self, op_name: str, op_type: str):
        self.op_name = op_name
        self.op_type = op_type
        self.param = {}
        self.inputs = []
        self.outputs = []
        self.has_host_inputs = False
        self.host_inputs = []
        self.has_reshape_inputs = False
        self.reshape_inputs = []
    
    def set_input(self, x):
        self.inputs = x
    
    def set_output(self, x):
        self.outputs = x
    
    def add_input(self, x):
        self.inputs.append(x)
    
    def add_output(self, x):
        self.outputs.append(x)
    
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
                "inputNames": self.inputs,
                "outputNames": self.outputs,
                "hasHostInputs": self.has_host_inputs,
                "hostInputNames": self.host_inputs,
                "hasReshapeInputs": self.has_reshape_inputs,
                "reshapeInputs": self.reshape_inputs,
            },
        }
        return node


class GetItemOperation(Operation):
    def __init__(self, name: str):
        super().__init__(name, "getitemOperation")
        self.index = -1


class InplaceOperation(Operation):
    def __init__(self, name: str):
        super().__init__(name, "inplaceOperation")
        self.input_index = -1
        self.target_index = -1
        self.target = None


class TupleOperation(Operation):
    def __init__(self, name: str):
        super().__init__(name, "tupleOperation")

class ViewOperation(Operation):
    def __init__(self, name):
        super().__init__(name, "viewOperation")
        self.target_shape = []
        self.target_reshape_info = {}


class GraphOpearation(Operation):
    def __init__(self, name: str):
        self.op_name = name
        self.op_type = "graphOperation"
        self.nodes = OrderedDict()
        self.inputs = []
        self.outputs = []
        self.internals = []
        self.node_size = -1
        self.node_names = []
        self.has_host_inputs = False
        self.host_inputs = []
        self.has_infer_shape = False
        self.infer_shape = ''
    
    def set_node_names(self, x):
        self.node_names = x
    
    def add_node_name(self, x):
        self.node_names.append(x)
    
    def set_inputs(self, x):
        self.inputs = x
    
    def set_outputs(self, x):
        self.outputs = x
    
    def set_internals(self, x):
        self.internals = x
    
    def set_nodes(self, x):
        self.nodes = x

    def add_input(self, x):
        self.inputs.append(x)
    
    def add_output(self, x):
        self.outputs.append(x)
    
    def add_internal(self, x):
        self.internals.append(x)
    
    def add_node(self, x):
        self.nodes.append(x)

    def build(self):
        graph = {
            "nodeType": "graphOperation",
            "value": {
            "nodes": [node.build() for _, node in self.nodes.items()],
            "inputNames": self.inputs,
            "outputNames": self.outputs,
            "internalNames": self.internals,
            "nodeSize": len(self.nodes),
            "hasHostInputs": self.has_host_inputs,
            "hostInputNames": self.host_inputs,
            "hasInferShape": self.has_infer_shape,
            "inferShape": self.infer_shape,
            }
        }
        return graph


class Graph:
    def __init__(self, name: str):
        self.name = name
        self.inputs = []
        self.outputs = []
        self.internals = []
        self.nodes = OrderedDict()
        self.hosts = []
    
    def set_hosts(self, x):
        self.hosts = x
    
    def set_inputs(self, x):
        self.inputs = x
        
    def set_outputs(self, x):
        self.outputs = x
    
    def set_internals(self, x):
        self.internals = x

    def add_input(self, x):
        self.inputs.append(x)
    
    def add_output(self, x):
        self.outputs.append(x)
    
    def add_internals(self, x):
        self.internals.append(x)
    
    def add_node(self, x):
        self.nodes[x.op_name] = x
    
    def to_json(self):
        atb_graph = {
            'name': self.name,
            'inputNames': self.inputs,
            'outputNames': self.outputs,
            'internalNames': self.internals,
            'nodes': [node for _, node in self.nodes.items()],
            'hostTensorNames': self.hosts,
            'nodeSize': len(self.nodes),
        }
        return json.dumps(atb_graph)

def get_input_data_node(node_list, node_name):
    for node in node_list:
        if node.name == node_name:
            return node
    return None

def make_output_tensor_desc(output_names,
                            output_data_nodes,
                            input_data_nodes,
                            graph_outputs,
                            inplace_tensor_dict,
                            inplace_tensor_with_shape_dict):
    output_tensor_descs = {'param': {}, 'create': {}}

    def process_node(output_name, node, input_name=None, need_reshape=False):
        dims, dim_num = get_shape(node)
        dims_prod = ' * '.join(dims)
        if node.name in inplace_tensor_with_shape_dict:
            target_shape = [str(x) for x in inplace_tensor_with_shape_dict[node.name]]
            if '-1' in target_shape:
                other_prod = ' * '.join(x for x in target_shape if x != '-1')
                negtive_idx = target_shape.index('-1')
                target_shape[negtive_idx] = f'({dims_prod}) // ({other_prod})'
            dims = target_shape
            dim_num = len(target_shape)
            need_reshape = True
        dims_str = f'[{",".join(dims)}]'
        dtype = get_dtype(node)
        info = f''' {{"format": {AclFormat.ACL_FORMAT_ND.value}, "dtype": {dtype}, "dimNum": {dim_num}, "dims": {dims_str} }} '''
        
        output_tensor_descs['param'][output_name] = info
        output_tensor_descs['create'][output_name] = {
            "dtype": str(get_torch_dtype(dtype)),
            "shape": dims_str,
            "input": input_name,
            "need_reshape": need_reshape,
        }

    for idx, output in enumerate(output_data_nodes):
        output_name = output_names[idx]
        if output_name in inplace_tensor_dict:
            input_name = inplace_tensor_dict[output_name]
            input_node = get_input_data_node(input_data_nodes, input_name)
            assert input_node is not None
            process_node(output_name, input_node, input_name)
        else:
            process_node(output_name, output)

    for output in graph_outputs:
        if output not in output_names:
            input_name = inplace_tensor_dict[output]
            input_node = get_input_data_node(input_data_nodes, input_name)
            assert input_node is not None
            process_node(output, input_node, input_name)

    return output_tensor_descs
    

def parse_graph(graph,
                input_names,
                output_names,
                input_data_nodes,
                output_data_nodes,
                py_output_names):
    ## define new graph 
    # inplace replace
    inplace_replace = {}
    inplace_tensor_to_real_tensor = {}
    for name in list(graph.nodes.keys()):
        node = graph.nodes[name]
        if node.op_type == "inplaceOperation":
            if node.input_index != -1:
                inplace_replace[node.outputs[0]] = f'{node.inputs[0]}__{node.input_index}'
            else:
                inplace_replace[node.outputs[0]] = node.inputs[0]
            if node.target_index != -1:
                inplace_tensor_to_real_tensor[inplace_replace[node.outputs[0]]] = f'{node.target}__{node.target_index}'
            else:
                inplace_tensor_to_real_tensor[inplace_replace[node.outputs[0]]] = node.target
            del graph.nodes[name]
    for name in graph.nodes.keys():
        node = graph.nodes[name]
        if not isinstance(node, GraphOpearation):
            for idx, input in enumerate(node.inputs):
                if input in inplace_replace.keys():
                    node.inputs[idx] = inplace_replace[input]
            for idx, output in enumerate(node.outputs):
                if output in inplace_replace.keys():
                    node.outputs[idx] = inplace_replace[output]
        else:
            for idx, output in enumerate(node.outputs):
                if output in inplace_replace.keys():
                    node.outputs[idx] = inplace_replace[output]

    # get tuple repalce
    tuple_replace = {}
    for name in list(graph.nodes.keys()):
        node = graph.nodes[name]
        if node.op_type == "tupleOperation":
            op_name = node.op_name
            for idx, input in enumerate(node.inputs):
                key_name = f'{op_name}__{idx}'
                tuple_replace[key_name] = input
            del graph.nodes[name]

    # get item replace
    getitem_replace = {}
    for name in list(graph.nodes.keys()):
        node = graph.nodes[name]
        if node.op_type == "getitemOperation":
            real_name = f'{node.inputs[0]}__{node.index}'
            if real_name in tuple_replace.keys():
                real_name = tuple_replace[real_name]
            getitem_replace[node.outputs[0]] = real_name
            del graph.nodes[name]
    for name in graph.nodes.keys():
        node = graph.nodes[name]
        if not isinstance(node, GraphOpearation):
            for idx, input in enumerate(node.inputs):
                if input in getitem_replace.keys():
                    node.inputs[idx] = getitem_replace[input]
        else:
            for idx, output in enumerate(node.outputs):
                if output in getitem_replace.keys():
                    node.outputs[idx] = getitem_replace[output]
    for idx, output in enumerate(output_names):
        if output in getitem_replace.keys():
            output_names[idx] = getitem_replace[output]

    # view replace
    view_replace = {}
    for name in list(graph.nodes.keys()):
        node = graph.nodes[name]
        if node.op_type == "viewOperation":
            view_replace[node.op_name] = node
            del graph.nodes[name]
    for name in graph.nodes.keys():
        node = graph.nodes[name]
        if not isinstance(node, GraphOpearation):
            need_reshape_input = False
            reshape_inputs = {}
            for idx, input in enumerate(node.inputs):
                if input in view_replace.keys():
                    reshape_info = view_replace[input].target_reshape_info
                    target_name = view_replace[input].inputs[0]
                    while target_name in view_replace.keys():
                        input = target_name
                        target_name = view_replace[target_name].inputs[0]
                    node.inputs[idx] = target_name
                    reshape_inputs[idx] = reshape_info
                    need_reshape_input = True
            node.has_reshape_inputs = need_reshape_input
            node.reshape_inputs = []
            if need_reshape_input:
                for idx, input in enumerate(node.inputs):
                    if idx in reshape_inputs.keys():
                        node.reshape_inputs.append(reshape_inputs[idx])
                    else:
                        node.reshape_inputs.append({"reshapeType": "None"})
        else:
            for idx, output in enumerate(node.outputs):
                if output in view_replace.keys():
                    target_name = view_replace[input].inputs[0]
                    while target_name in view_replace.keys():
                        input = target_name
                        target_name = view_replace[target_name].inputs[0]
                    node.outputs[idx] = target_name             
    view_replace_name_dict = {}
    for k, v in view_replace.items():
        view_replace_name_dict[k] = v.inputs[0]

    # graph operation
    graph_nodes = []
    for _, node in graph.nodes.items():
        if node.op_type == "graphOperation":
            graph_nodes.append(node)
    for graph_node in graph_nodes:
        graph_inputs = []
        graph_outputs = []
        graph_hosts = []
        graph_internals = []
        for node_name in graph_node.node_names:
            if node_name not in graph.nodes.keys():
                continue
            graph_node.nodes[node_name] = graph.nodes[node_name]
            del graph.nodes[node_name]
            node = graph_node.nodes[node_name]
            for input in node.inputs:
                graph_inputs.append(input)
            for output in node.outputs:
                graph_outputs.append(output)
            if node.has_host_inputs:
                graph_hosts.extend(node.host_inputs)

        graph_inputs = list(set(graph_inputs))
        graph_outputs = list(set(graph_outputs))
        graph_internals = list(set(graph_internals))
        graph_hosts = list(set(graph_hosts))

        graph_inputs = [x for x in graph_inputs if x not in graph_outputs]
        for k, v in inplace_tensor_to_real_tensor.items():
            v_in_input = v in graph_inputs
            if not v_in_input:
                while v in view_replace_name_dict.keys() and not v_in_input:
                    v = view_replace_name_dict[v]
                    v_in_input = v in graph_inputs
            if v_in_input and k not in graph_node.outputs:
                graph_node.outputs.append(k)

        graph_internals = [x for x in graph_outputs if x not in graph_inputs and x not in graph_node.outputs]

        graph_node.set_inputs(graph_inputs)
        graph_node.set_internals(graph_internals)
        # graph_node.set_internals([])
        # # import pdb;pdb.set_trace()
        # graph_internals.extend(graph_node.outputs)
        # graph_node.set_outputs(list(set(graph_internals)))
        graph_node.node_names = list(graph_node.nodes.keys())
        if graph_node.has_infer_shape:
            infer_shape = []
            for item in graph_node.infer_shape["value"]:
                node_id, tensor_id = item
                node_name = graph_node.node_names[node_id]
                input_name = graph_node.nodes[node_name].inputs[tensor_id]
                infer_shape.append(graph_node.inputs.index(input_name))
            graph_node.infer_shape = {'type': 'equal', 'value': infer_shape}
        if len(graph_hosts) > 0:
            graph_node.has_host_inputs = True
            graph_node.host_inputs = graph_hosts

    ## run graph
    inplace_tensor_with_reshape = {}
    for inplace_tensor, target_tensor in inplace_tensor_to_real_tensor.items():
        if target_tensor in getitem_replace.keys():
            target_tensor = getitem_replace[target_tensor]
        if target_tensor in view_replace.keys():
            target_shape = view_replace[target_tensor].target_shape
            while target_tensor in view_replace.keys():
                target_tensor = view_replace[target_tensor].inputs[0]
            inplace_tensor_with_reshape[target_tensor] = target_shape
        inplace_tensor_to_real_tensor[inplace_tensor] = target_tensor


    all_tensors = []
    host_tensors = []
    for name, node in graph.nodes.items():
        all_tensors.extend(node.inputs)
        all_tensors.extend(node.outputs)
        if node.has_host_inputs:
            host_tensors.extend(node.host_inputs)
    all_tensors = list(set(all_tensors))
    host_tensors = list(set(host_tensors))

    node_count = 0
    node_hosts = []
    node_inputs = []
    node_inputs_count = {}
    for input in input_names:
        node_inputs_count[input] = 0
    
    for name, node in graph.nodes.items():
        for ti, t_name in enumerate(node.inputs):
            if t_name in host_tensors and node.has_host_inputs:
                node_hosts.append({"nodeId": node_count, "tensorId": ti, "tensorName": t_name})
            # elif t_name in node_inputs_count.keys():
            #     node_inputs_count[t_name] = node_inputs_count[t_name] + 1
            if t_name in node_inputs_count.keys():
                node_inputs_count[t_name] = node_inputs_count[t_name] + 1
        node_count = node_count + 1
    # import pdb;pdb.set_trace()
    for tensor, count in node_inputs_count.items():
        if count > 0:
            node_inputs.append(tensor)
    
    node_outputs = copy.deepcopy(output_names)
    node_internals = []
    for k, v in inplace_tensor_to_real_tensor.items():
        if v in node_inputs and k not in node_outputs:
            node_outputs.append(k)
    for tensor in all_tensors:
        if tensor not in node_inputs and tensor not in node_outputs:
            node_internals.append(tensor)
    # print('node_inputs:', node_inputs)
    # print('node_outputs:', node_outputs)
    # print('node_internals:', node_internals)
    
    graph.set_inputs(node_inputs)
    graph.set_outputs(node_outputs)
    graph.set_internals(node_internals)
    graph.set_hosts(node_hosts)
    for name in graph.nodes.keys():
        graph.nodes[name] = graph.nodes[name].build()

    output_tensor_descs = make_output_tensor_desc(output_names,
                                                  output_data_nodes,
                                                  input_data_nodes,
                                                  node_outputs,
                                                  inplace_tensor_to_real_tensor,
                                                  inplace_tensor_with_reshape)

    for idx, tensor in enumerate(py_output_names):
        if tensor in getitem_replace.keys():
            py_output_names[idx] = getitem_replace[tensor]

    return graph, output_tensor_descs, py_output_names


