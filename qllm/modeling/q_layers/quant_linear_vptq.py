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

    def __init__(self, bits, groupsize, infeatures, outfeatures, bias):
        super().__init__()
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
        self.register_buffer('scales', torch.zeros(
            (math.ceil(infeatures / self.groupsize), outfeatures), dtype=torch.float16))
        self.register_buffer('g_idx', torch.tensor([i // self.groupsize for i in range(infeatures)], dtype=torch.int32))
        if bias:
            self.register_buffer('bias', torch.zeros((outfeatures), dtype=torch.float16))
        else:
            self.bias = None

    def handle_qzeros_for_autogptq(self):
        if self.qzeros.numel() == 0:
            return
        device = "cuda" if torch.cuda.is_available() else "cpu"
        qzeros = self.qzeros.to(device)
        zeros = torch.zeros((self.outfeatures, self.infeatures // self.groupsize),
                            dtype=torch.int32, device=qzeros.device).T.contiguous()

        general_unpack_on_row(qzeros, zeros, self.bits)

        zeros += 1
        torch.bitwise_and(zeros, (2 ** self.bits) - 1, out=zeros)

        general_pack_on_row(qzeros, zeros, self.bits)

        self.qzeros = qzeros.to("cpu", non_blocking=True)

    def forward(self, x):
        if self.act_order is None:
            self.act_order = self.g_idx[:self.groupsize].sum() != 0
        g_idx = self.g_idx if self.act_order else None
        out = QuantLinearTorchFunction_forward(x, self.qweight, self.scales,
                                               self.qzeros, g_idx, self.bits, self.groupsize, self.infeatures)
        out = out + self.bias if self.bias is not None else out
        return out
