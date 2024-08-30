import torch
from dicp.dynamo_bridge.compile_fx import is_torch_210
from dicp.vendor.AtbGraph.conversion import AtenToAtbTransformer
from ...dynamo_bridge.graph import GraphTransformer

if is_torch_210:
    from dicp.dynamo_bridge.op_transformer import BackendPatternMatcherTransformer
    from dicp.vendor.AtbGraph.pattern_replacement import (
        atb_pattern_matcher,
        torch_patterns_cls_list_1,
        torch_patterns_cls_list_2,
        torch_patterns_cls_list_3,
    )


def atbgraph_opset_convert(
    gm: torch.fx.GraphModule,
):
    # gm.print_readable()
    # import pdb;pdb.set_trace()
    gm = BackendPatternMatcherTransformer(
        atb_pattern_matcher, torch_patterns_cls_list_3).transform(gm)
    gm.print_readable()
    # import pdb;pdb.set_trace()

    # gm = BackendPatternMatcherTransformer(
    #     atb_pattern_matcher, torch_patterns_cls_list_1).transform(gm)


    gm = AtenToAtbTransformer(gm).transform()

    # For bug in pytorch
    # Avoid for dynamic shape
    gt = GraphTransformer(gm, "atbgraph")
    gt.infer_shape_dtype()
    gm = gt.gm
    return gm


