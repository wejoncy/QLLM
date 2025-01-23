import math
import os

import torch
import torch.nn as nn
from .ext_package_checker import has_ort_ops
from .compress_weight import (CompressWeight, general_pack_on_row,
                              general_unpack_on_row)
if has_ort_ops():
    from qllm import ort_ops


# fake_layer
class VQuantLinear(nn.Module, CompressWeight):

    def __init__(self, bits, groupsize, infeatures, outfeatures, bias, dtype=None):
        super().__init__()
        self.dtype = torch.get_default_dtype() if dtype is None else dtype
        if bits not in [2, 3, 4, 5, 6, 7, 8]:
            raise NotImplementedError("Only 2,4,5,6,7,8 bits are supported.")
        self.infeatures = infeatures
        self.outfeatures = outfeatures
        self.bits = bits
        self.act_order = None
        self.orig_fp_weight = None
        self.maxq = 2**self.bits - 1
        self.groupsize = groupsize if groupsize != -1 else infeatures
        self.pack_mode = "GPTQ"

        self.register_buffer('qweight', torch.zeros((infeatures // 32 * self.bits, outfeatures), dtype=torch.int32))
        self.register_buffer('qzeros', torch.zeros((math.ceil(infeatures / self.groupsize),
                             outfeatures // 32 * self.bits), dtype=torch.int32))
        self.register_buffer("scales", torch.zeros((math.ceil(infeatures / self.groupsize), outfeatures), dtype=self.dtype))
        self.register_buffer('g_idx', torch.tensor([i // self.groupsize for i in range(infeatures)], dtype=torch.int32))
        if bias:
            self.register_buffer("bias", torch.zeros((outfeatures), dtype=self.dtype))
        else:
            self.bias = None

    
    def forward(self, x):
        return x
