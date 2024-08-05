import os
import subprocess
import time

import torch
import dicp
from dicp.dynamo_bridge.compile import DeviceCompileJob
from torch._inductor.codecache import pick_vec_isa, cpp_compile_command, write, code_hash
from torch._inductor import exc

from dicp.vendor.AtbGraph.codegen import load_and_run

class AtbCompileJob(DeviceCompileJob):
    def __init__(self, source_code) -> None:
        super().__init__()
        picked_vec_isa = pick_vec_isa()
        self._local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self._key, self._input_path = write(
            source_code.strip(),
            "json",
            extra=cpp_compile_command("i", "o", vec_isa=picked_vec_isa) +
            'local_rank' + str(self._local_rank)
        )
        self._output_graph_path = self._input_path[:-5] + '/graph'


    def _compile(self):
        try:
            if not hasattr(torch.classes.ModelTorch, "ModelTorch"):
                torch.classes.load_library('/tzy/dicp_atb_new/output/atb_speed/lib/libatb_speed_torch.so')
        except Exception as e:
            torch.classes.load_library('/tzy/dicp_atb_new/output/atb_speed/lib/libatb_speed_torch.so')

    def get_key(self):
        return self._key

    def get_compile_result(self):
        self._compile()
        return load_and_run.AtbModel(self._input_path)
