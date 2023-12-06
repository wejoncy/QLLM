import math
import os
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import custom_bwd, custom_fwd
import importlib
from .compress_weight import CompressWeight, general_unpack_on_row, general_pack_on_row


try:
    has_module_XbitOps = importlib.util.find_spec("XbitOps")

    if not has_module_XbitOps:
        import subprocess
        import sys
        subprocess.check_call([sys.executable, '-m', 'pip', 'install',
                              'git+https://github.com/wejoncy/XbitOps.git'])
        import XbitOps  # cuda dequant
        has_module_XbitOps = True
    else:
        import XbitOps  # cuda dequant
except:
    print("torch implementation of dequantization would be used")
    pass


class QuantLinearTorchFunction(torch.autograd.Function):
    @staticmethod
    def symbolic(g,  x, qself_qweight, qself_scales, qself_qzeros, qself_sbias, qself_g_idx, bits, groupsize, in_features):
        return g.op("com.microsoft::QuantNbitsGemm", x, qself_qweight, qself_scales, qself_qzeros, qself_sbias, qself_g_idx,
                    outputs=1, in_features_i=in_features, bits_i=bits, groupsize_i=groupsize)

    @staticmethod
    def forward(ctx, input, qweight, scales, qzeros, sbias, g_idx, bits, groupsize, in_features):
        if torch.onnx.is_in_onnx_export():
            return torch.zeros(input.shape[:-1] + (qweight.size(1), ), dtype=input.dtype, device=input.device)

        wf = torch.tensor(list(range(0, 32, bits)), dtype=torch.int32, device=input.device).unsqueeze(0)
        zeros = torch.bitwise_right_shift(torch.unsqueeze(qzeros, 2), wf.unsqueeze(0)
                                          ).to(torch.int16 if bits == 8 else torch.int8)
        zeros = torch.bitwise_and(zeros, (2 ** bits) - 1)

        # zeros = zeros + 1
        zeros = zeros.reshape(-1, 1, zeros.shape[1] * zeros.shape[2])

        scales = scales.reshape(-1, 1, scales.shape[-1])

        weight = torch.bitwise_right_shift(torch.unsqueeze(qweight, 1), wf.unsqueeze(-1)
                                           ).to(torch.int16 if bits == 8 else torch.int8)
        torch.bitwise_and(weight, (2 ** bits) - 1, out=weight)
        weight = weight.reshape(-1, groupsize, weight.shape[2])

        # if g_idx is not sorted
        # sz=scale_zeros[g_idx.long()].squeeze(1)
        # ss=scales[g_idx.long()].squeeze(1)
        # weight=weight.reshape(-1,12288)*ss-sz

        scale_zeros = zeros * scales
        weight = (scales * weight - scale_zeros.half())

        # weight = (scales * (weight - zeros))
        weight = weight.reshape(-1, weight.shape[2])
        out = torch.matmul(input, weight.contiguous())
        return out



class DequantAndUnpack(torch.autograd.Function):
    @staticmethod
    def symbolic(g, qself_qweight, scales, qzeros, groupsize, bits, in_features, g_idx, act_order):
        return g.op("com.microsoft::DequantizeAndUnpackWeight", qself_qweight, scales, qzeros,
                    outputs=1, groupsize_i=groupsize, bits_i=bits, in_features_i=in_features)

    @staticmethod
    def forward(ctx, qweight, scales, qzeros, groupsize, bits, in_features, g_idx, act_order):
        load_from_autogptq = int(os.environ.get('load_from_autogptq', "0"))
        if has_module_XbitOps and qweight.is_cuda and not act_order:
            return XbitOps.dequant(qweight, scales, qzeros, groupsize, bits, in_features, load_from_autogptq)
        scales = scales.reshape(-1, 1, scales.shape[-1])
        if bits in [2, 4, 8]:
            wf = torch.tensor(list(range(0, 32, bits)), dtype=torch.int32, device=qweight.device).unsqueeze(0)
            # expand is removed as torch will auto broadcast to relavant dimension
            zeros = torch.bitwise_right_shift(torch.unsqueeze(qzeros, 2), wf.unsqueeze(0)
                                              ).to(torch.int16 if bits == 8 else torch.int8)
            zeros = torch.bitwise_and(zeros, (2 ** bits) - 1)
            zeros = zeros + load_from_autogptq
            zeros = zeros.reshape(-1, 1, zeros.shape[1] * zeros.shape[2])
            # expand is removed as torch will auto broadcast to relavant dimension
            weight = torch.bitwise_right_shift(torch.unsqueeze(
                qweight, 1), wf.unsqueeze(-1)).to(torch.int16 if bits == 8 else torch.int8)
            torch.bitwise_and(weight, (2 ** bits) - 1, out=weight)
        else:
            weight = torch.zeros((in_features, qweight.shape[1]), dtype=torch.int32, device=qweight.device)
            general_unpack_on_row(qweight, weight, bits)
            zeros = torch.zeros((qzeros.shape[0], qweight.shape[1]), dtype=torch.int32, device=qweight.device)
            zeros = zeros.T
            general_unpack_on_row(qzeros.T, zeros, bits)
            zeros = zeros.T
            zeros = zeros.reshape(-1, 1, zeros.shape[1])

        if act_order:
            zeros.squeeze_(1)
            scales.squeeze_(1)
            weight = weight.view(-1, weight.shape[-1])
            scale_zeros = zeros * scales
            weight = (scales[g_idx] * weight - scale_zeros[g_idx])
            weight = weight.view(-1, groupsize, weight.shape[-1])
        else:
            scale_zeros = zeros * scales
            weight = weight.reshape(-1, groupsize, weight.shape[-1])
            weight = (scales * weight - scale_zeros.half())

        # weight = (scales * (weight - zeros))
        weight = weight.reshape(-1, weight.shape[2])
        return weight


def QuantLinearTorchFunction_forward(input, qweight, scales, qzeros, g_idx, bits, groupsize, in_features, act_order):
    load_from_autogptq = int(os.environ.get('load_from_autogptq', "0"))
    if not act_order and not torch.onnx.is_in_onnx_export() and input.reshape(-1, input.shape[-1]).shape[0] <= 4 and bits == 4 and groupsize==128:
        return XbitOps.gemv(input, qweight, scales, qzeros, groupsize, bits, in_features, load_from_autogptq)
    weight = DequantAndUnpack().apply(qweight, scales, qzeros, groupsize, bits, in_features, g_idx, act_order)
    out = torch.matmul(input, weight.contiguous())
    return out


class QuantLinear(nn.Module, CompressWeight):

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
        device = "cuda" if torch.cuda.is_available() else "cpu"
        qzeros = self.qzeros.T.contiguous().to(device)
        zeros = torch.zeros((self.outfeatures, self.infeatures//self.groupsize),
                            dtype=torch.int32, device=qzeros.device)

        general_unpack_on_row(qzeros, zeros, self.bits)
        zeros = zeros.reshape(-1, zeros.shape[-1]).to(torch.int32)

        zeros += 1

        general_pack_on_row(qzeros, zeros, self.bits)

        self.qzeros = qzeros.T.contiguous().cpu()


    def forward(self, x):
        if self.act_order is None:
            self.act_order = not (self.g_idx[:self.groupsize//self.bits].sum() == 0)
        # would contiguous() here affect accuracy? the default should be contiguous().T
        # out =  torch.matmul(x.reshape(-1, x.shape[-1]),self.orig_fp_weight.T.contiguous())
        out_shape = x.shape[:-1] + (self.outfeatures, )
        # out = QuantLinearFunction.apply(x.reshape(-1, x.shape[-1]), self.qweight, self.scales, self.qzeros, self.g_idx, self.bits, self.maxq)
        # sbias = self.bias if self.bias is not None else torch.tensor([0],dtype=torch.float16)
        # out = QuantLinearTorchFunction.apply(x, self.qweight, self.scales, self.qzeros, sbias, self.g_idx, self.bits, self.groupsize, self.infeatures)
        out = QuantLinearTorchFunction_forward(x, self.qweight, self.scales,
                                               self.qzeros, self.g_idx, self.bits, self.groupsize, self.infeatures, self.act_order)
        out = out + self.bias if self.bias is not None else out
        return out
