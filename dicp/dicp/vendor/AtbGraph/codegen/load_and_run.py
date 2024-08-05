import atexit
import os

import acl
import numpy as np
import torch
from torch.profiler import record_function


class AtbModel():
    def __init__(self, model_path) -> None:
        self.model = torch.classes.ModelTorch.ModelTorch("dicp_DICPCustomModel")
        self.model.set_param(model_path)

    def run(self, inputs, outputs, param):
        # inputs = [x.npu() for x in inputs]
        if len(outputs) > 0:
            try:
                o =  self.model.execute_out(inputs, outputs, param)
            except Exception as e:
                import pdb;pdb.set_trace()
                pass
            return o
        else:
            return self.model.execute(inputs, param)


if __name__ == '__main__':
    pass