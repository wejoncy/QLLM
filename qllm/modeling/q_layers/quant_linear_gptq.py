import math
import os

import torch
import torch.nn as nn
from .ext_package_checker import has_ort_ops
from .compress_weight import (CompressWeight, general_pack_on_row,
                              general_unpack_on_row)
if has_ort_ops():
    from qllm import ort_ops


def DequantizeLinearBlockWise(qweight, scales, qzeros, groupsize, bits, in_features, g_idx):
    COMPATIBLE_WITH_AUTOGPTQ = int(
        os.environ.get("COMPATIBLE_WITH_AUTOGPTQ", "0"))
    scales = scales.reshape(-1, 1, scales.shape[-1])
    if bits in [2, 4, 8]:
        wf = torch.tensor(list(range(0, 32, bits)), dtype=torch.int32, device=qweight.device).unsqueeze(0)
        # expand is removed as torch will auto broadcast to relavant dimension
        zeros = torch.bitwise_right_shift(torch.unsqueeze(qzeros, 2), wf.unsqueeze(0)
                                            ).to(torch.int16 if bits == 8 else torch.int8)
        zeros = zeros + COMPATIBLE_WITH_AUTOGPTQ
        zeros = torch.bitwise_and(zeros, (2 ** bits) - 1)
        zeros = zeros.reshape(-1, 1, zeros.shape[1] * zeros.shape[2])
        # expand is removed as torch will auto broadcast to relavant dimension
        weight = torch.bitwise_right_shift(torch.unsqueeze(
            qweight, 1), wf.unsqueeze(-1)).to(torch.int16 if bits == 8 else torch.int8)
        torch.bitwise_and(weight, (2 ** bits) - 1, out=weight)
    else:
        weight = torch.zeros((in_features, qweight.shape[1]), dtype=torch.int32, device=qweight.device)
        general_unpack_on_row(qweight, weight, bits)
        zeros = torch.zeros((qzeros.shape[0], qweight.shape[1]), dtype=torch.int32, device=qweight.device)
        general_unpack_on_row(qzeros, zeros, bits)
        zeros = zeros.reshape(-1, 1, zeros.shape[1])
        zeros = zeros + COMPATIBLE_WITH_AUTOGPTQ
        zeros = torch.bitwise_and(zeros, (2 ** bits) - 1)

    if g_idx is not None:
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


class QuantLinearTorchFunction(torch.autograd.Function):
    @staticmethod
    def symbolic(g, inputs, qweight, scales, qzeros, groupsize, bits, in_features, g_idx):
        bias = g.op("Constant", value_t=torch.tensor([], dtype=torch.float16))
        g_idx = g.op("Constant", value_t=torch.tensor([], dtype=torch.int32)) if g_idx is None else g_idx
        use_gemm_op = True
        if use_gemm_op:
            out_features = qweight.type().sizes()[-1]
            return g.op("com.microsoft::MatMulNBits", inputs, qweight, scales, qzeros, g_idx,
                        outputs=1, K_i=in_features, N_i=out_features, bits_i=bits, block_size_i=groupsize, packing_s="gptq")
        else:
            fp_weight = g.op("com.microsoft::DequantizeLinearBlockWise", qweight, scales, qzeros, g_idx,
                             outputs=1, groupsize_i=groupsize, bits_i=bits, in_features_i=in_features)
            return g.op("MatMul", inputs, fp_weight)

    @staticmethod
    def forward(ctx, inputs, qweight, scales, qzeros, groupsize, bits, in_features, g_idx):
        if torch.onnx.is_in_onnx_export():
            return torch.zeros(inputs.shape[:-1] + (qweight.size(1), ), dtype=inputs.dtype, device=inputs.device)

        COMPATIBLE_WITH_AUTOGPTQ = int(os.environ.get("COMPATIBLE_WITH_AUTOGPTQ", "0"))
        if (not torch.onnx.is_in_onnx_export()
            and inputs.numel() // inputs.shape[-1] <= 8
            and bits == 4
                and has_ort_ops()):
            return ort_ops.gemv(inputs, qweight, scales, qzeros, g_idx, groupsize, bits, in_features, COMPATIBLE_WITH_AUTOGPTQ)
        if qweight.is_cuda and has_ort_ops():
            weight = ort_ops.dequant(qweight, scales, qzeros, g_idx, groupsize, bits, in_features, COMPATIBLE_WITH_AUTOGPTQ)
        else:
            weight = DequantizeLinearBlockWise(qweight, scales, qzeros, groupsize, bits, in_features, g_idx)
        return torch.matmul(inputs, weight)


def QuantLinearTorchFunction_forward(input, qweight, scales, qzeros, g_idx, bits, groupsize, in_features):
    return QuantLinearTorchFunction().apply(input, qweight, scales, qzeros, groupsize, bits, in_features, g_idx)


class QuantLinearGPTQ(nn.Module, CompressWeight):

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
