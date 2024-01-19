import math
import torch
import torch.nn as nn

from .compress_weight import (CompressWeight, general_unpack_on_row)


class DequantAndUnpack(torch.autograd.Function):
    @staticmethod
    def forward(ctx, qweight, scales, qzeros, groupsize, bits, in_features):
        scales = scales.reshape(-1, 1, scales.shape[-1])
        qzeros = qzeros.reshape(-1, 1, qzeros.shape[-1])
        if bits in [2, 4, 8]:
            wf = torch.tensor(list(range(0, 32, bits)), dtype=torch.int32, device=qweight.device).unsqueeze(0)
            weight = torch.bitwise_right_shift(torch.unsqueeze(
                qweight, 1), wf.unsqueeze(-1)).to(torch.int16 if bits == 8 else torch.int8)
            torch.bitwise_and(weight, (2 ** bits) - 1, out=weight)
        else:
            weight = torch.zeros((in_features, qweight.shape[1]), dtype=torch.int32, device=qweight.device)
            general_unpack_on_row(qweight, weight, bits)

        scale_zeros = qzeros * scales
        weight = weight.reshape(-1, groupsize, weight.shape[-1])
        weight = (scales * weight - scale_zeros.half())

        # weight = (scales * (weight - zeros))
        weight = weight.reshape(-1, weight.shape[2])
        return weight


class QuantLinearTorchFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, qweight, scales, qzeros, groupsize, bits, in_features):
        if torch.onnx.is_in_onnx_export():
            return torch.zeros(inputs.shape[:-1] + (qweight.size(1), ), dtype=inputs.dtype, device=inputs.device)

        weight = DequantAndUnpack.apply(qweight, scales, qzeros, groupsize, bits, in_features)
        return torch.matmul(inputs, weight)

    @staticmethod
    def symbolic(g, inputs, qweight, scales, qzeros, groupsize, bits, in_features):
        out_features = qweight.type().sizes()[-1]
        return g.op("com.microsoft::MatMulNBits", inputs, qweight, scales, qzeros, 
                    outputs=1, K_i=in_features, N_i=out_features, bits_i=bits, block_size_i=groupsize, packing_s="hqq")


class QuantLinearHQQ(nn.Module, CompressWeight):

    def __init__(self, bits, groupsize, infeatures, outfeatures, bias):
        super().__init__()
        if bits not in [2, 3, 4, 5, 6, 7, 8]:
            raise NotImplementedError("Only 2,4,5,6,7,8 bits are supported.")
        self.infeatures = infeatures
        self.outfeatures = outfeatures
        self.bits = bits
        self.groupsize = groupsize if groupsize != -1 else infeatures
        self.pack_mode = "HQQ"
        self.orig_fp_weight = None
        self.g_idx = torch.tensor([i // groupsize for i in range(infeatures)], dtype=torch.int32)

        self.register_buffer('qweight', torch.zeros((infeatures // 32 * self.bits, outfeatures), dtype=torch.int32))
        self.register_buffer('qzeros', torch.zeros((math.ceil(infeatures / self.groupsize),
                             outfeatures), dtype=torch.float16))
        self.register_buffer('scales', torch.zeros(
            (math.ceil(infeatures / self.groupsize), outfeatures), dtype=torch.float16))
        if bias:
            self.register_buffer('bias', torch.zeros((outfeatures), dtype=torch.float16))
        else:
            self.bias = None

    def pack_qzeros_even(self, intzeros, device):
        self.qzeros = intzeros.contiguous().cpu()

    def pack_qzeros_odd(self, intzeros, device):
        self.qzeros = intzeros.contiguous().cpu()

    def forward(self, x):
        out = QuantLinearTorchFunction.apply(x, self.qweight, self.scales, self.qzeros,
                                             self.groupsize, self.bits, self.infeatures)
        out = out + self.bias if self.bias is not None else out
        return out
