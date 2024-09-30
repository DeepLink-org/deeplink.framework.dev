import torch
import torch.fx
import os

from dicp.vendor.TopsGraph.conversion import AtenToTopsTransformer
from dicp.vendor.TopsGraph.to_clast import TopsMemoryFormatTransformer
from dicp.dynamo_bridge.torch_version import is_torch_210_or_higher

if is_torch_210_or_higher:
    from dicp.dynamo_bridge.op_transformer import BackendPatternMatcherTransformer
    from dicp.vendor.TopsGraph.conversion import tops_patterns, tops_patterns_cls_list


class HandleInplaceCopyPass():
    def transform(self, gm: torch.fx.GraphModule):
        nodes = list(gm.graph.nodes)
        last_node = nodes[-1]
        assert last_node.op == "output"

        origin_outputs = list(last_node.args[0])
        inplace_outputs = []
        inplace_dict = {}
        for node in reversed(nodes):
            if node.op not in ["placeholder", "output"] and not isinstance(node.target, str):
                if hasattr(node.target, "name") and node.target.name() == "Copy_":
                    if node.args[0].op == "placeholder" and node.args[0].name not in inplace_dict.values():
                        inplace_outputs.append(node.args[1])
                        inplace_dict[node.args[1].name] = node.args[0].name

        assert len(last_node.kwargs) == 0
        last_node._Node__update_args_kwargs((origin_outputs + inplace_outputs, ), inplace_dict)

        return gm


def topsgraph_opset_transform(
    gm: torch.fx.GraphModule,
):

    # 1aten to Ntops
    gm = AtenToTopsTransformer(gm).transform()

    # Ntops to Mtops
    if is_torch_210_or_higher:
        gm = BackendPatternMatcherTransformer(
            tops_patterns, tops_patterns_cls_list).transform(gm)

    # tops: normal shape to clast shape
    gm = TopsMemoryFormatTransformer().transform(gm)

    # handle inplace copy operation: get inplace copy args to update outputs.
    gm = HandleInplaceCopyPass().transform(gm)

    return gm
