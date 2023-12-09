import math
import os
from importlib.util import find_spec

import numpy as np
import torch
import torch.nn as nn

from .compress_weight import CompressWeight, general_unpack_on_row
import ort_ops

DEBUG_ = False
class QuantLinearTorchFunction(torch.autograd.Function):
    @staticmethod
    def symbolic(g,  x, qself_qweight, qself_scales, qself_qzeros, bits, groupsize, in_features, out_features):
        return g.op("com.microsoft::MatMulNBits", x, qself_qweight, qself_scales, qself_qzeros,
                    outputs=1, K_i=in_features, N_i=out_features, bits_i=bits, block_size_i=groupsize)

    @staticmethod
    def forward(ctx, x, qself_qweight, qself_scales, qself_qzeros, bits, groupsize, in_features, out_features):
        if torch.onnx.is_in_onnx_export():
            return torch.zeros(x.shape[:-1] + (out_features, ), dtype=x.dtype, device=x.device)
        fp_weight = ort_ops.Dequantize4Bits(
            qself_qweight, qself_qzeros, qself_scales, groupsize, in_features, out_features)
        return torch.matmul(x, fp_weight.T)


def QuantLinearTorchFunction_forward(input, qweight, scales, qzeros, bits, groupsize, in_features, out_features):
    assert bits==4, "Only 4 bits are supported."
    out = QuantLinearTorchFunction().apply(
        input, qweight, scales, qzeros, bits, groupsize,  in_features, out_features)
    return out


def dequantize_blockwise_4bits(quant_values, scale, zero_point, rows, cols):
    expand_quant_value = (np.repeat(quant_values, 2, -1).reshape(*quant_values.shape, 2) >> [0, 4]) & 0x0F
    expand_quant_value = expand_quant_value.reshape(*quant_values.shape[:-1], -1)
    aligned_scale = scale.reshape(*quant_values.shape[:-1], 1)
    expand_zero_point = (np.repeat(zero_point, 2, -1).reshape(-1, 2) >> [0, 4]) & 0xF
    expand_zero_point = expand_zero_point.reshape(-1)
    if (quant_values.size // quant_values.shape[-1]) & 1:
        expand_zero_point = expand_zero_point[:-1]
    expand_zero_point = expand_zero_point.reshape(*quant_values.shape[:-1], -1)
    float_values = ((expand_quant_value - expand_zero_point) * aligned_scale).astype(scale.dtype)
    float_values = float_values.reshape(cols, -1)
    float_values = float_values[:, :rows]
    return float_values, expand_zero_point, aligned_scale

class QuantLinearORT(nn.Module, CompressWeight):
    def __init__(self, bits, groupsize, infeatures, outfeatures, bias):
        super().__init__()
        if bits not in [2, 3, 4, 5, 6, 7, 8]:
            raise NotImplementedError("Only 2,4,5,6,7,8 bits are supported.")
        self.infeatures = infeatures
        self.outfeatures = outfeatures
        self.bits = bits
        self.orig_fp_weight = None
        self.maxq = 2**self.bits - 1
        self.groupsize = groupsize if groupsize != -1 else infeatures
        self.act_order = None
        self.pack_mode = "ORT"

        self.register_buffer('qweight', torch.zeros(
            (outfeatures, infeatures//self.groupsize, self.groupsize//(8//bits)), dtype=torch.uint8))
        self.register_buffer('qzeros', torch.zeros((math.ceil(infeatures // self.groupsize)*(
                             outfeatures // 8 * self.bits)), dtype=torch.uint8))
        self.register_buffer('scales', torch.zeros(
            (math.ceil(infeatures / self.groupsize) * outfeatures), dtype=torch.float16))
        self.register_buffer('g_idx', torch.tensor(
            [i // self.groupsize for i in range(infeatures)], dtype=torch.int32))
        if bias:
            self.register_buffer('bias', torch.zeros(
                (outfeatures), dtype=torch.float16))
        else:
            self.bias = None

    def pack_on_device_for_even_bits(self, intweight_gpu, intzeros_T):
        zero_point_unpacked = intzeros_T.T.cpu().numpy().astype(np.uint8)
        matrix_int8 = intweight_gpu.cpu().numpy().astype(np.uint8)
        scales = self.scales.T.cpu().numpy()
        block_size = self.groupsize
        
        # Reference dequant
        #mm = ((matrix_int8.T.reshape(matrix_int8.shape[-1], -1, block_size).astype(np.int64) -
        #      zero_point_unpacked[:, :, None].astype(np.int64))*scales[:, :, None]).astype(scales.dtype
        #      ).reshape(matrix_int8.shape[-1], -1)

        rows, cols = matrix_int8.shape
        blob_size = block_size // 2
        k_blocks = (rows + block_size - 1) // block_size
        padded_rows = k_blocks * block_size
        pad_len = padded_rows - rows
        matrix_int8_padded = matrix_int8
        if pad_len > 0:
            matrix_int8_padded = np.pad(
                matrix_int8, ((0, pad_len), (0, 0)), "constant")

        matrix_int8_padded = np.transpose(matrix_int8_padded)
        int8_values = matrix_int8_padded

        zero_point = zero_point_unpacked.reshape(-1)
        zero_point = np.concatenate((zero_point, np.array([8], dtype=np.uint8))) if zero_point.shape[0] & 1 else zero_point
        zero_point = zero_point[0::2] | (zero_point[1::2] << 4)
        zero_point = zero_point.reshape(-1)

        packed = (int8_values[:, 0::2]) | (int8_values[:, 1::2] << 4)
        packed = packed.reshape(cols, k_blocks, blob_size)
        scales = scales.reshape(-1)

        assert self.qweight.shape == packed.shape
        assert self.qzeros.shape == zero_point.shape

        self.scales = torch.from_numpy(scales)
        self.qweight = torch.from_numpy(packed)
        self.qzeros = torch.from_numpy(zero_point)

        if DEBUG_:
            mat_float,_,_ = dequantize_blockwise_4bits(packed, scales, zero_point, rows, cols)
            print('mat_float', mat_float.shape, mat_float.dtype)

    def unpack(self):
        quant_values, scale, zero_point, rows, cols = (self.qweight.cpu().numpy(
        ), self.scales.cpu().numpy(), self.qzeros.cpu().numpy(), self.infeatures, self.outfeatures)
        
        float_values, zero_point, scale = dequantize_blockwise_4bits(quant_values, scale, zero_point, rows, cols)
        float_values = torch.from_numpy(float_values.T)
        zero_point = torch.from_numpy(zero_point.T)
        scale = torch.from_numpy(scale.T)
        return float_values, zero_point, scale

    def forward(self, x):
        if self.act_order is None:
            self.act_order = not (self.g_idx[:self.groupsize//self.bits].sum() == 0)
            assert not self.act_order, "onnxruntime doesn't support g_idx for now."
        out = QuantLinearTorchFunction_forward(x, self.qweight, self.scales,
                                               self.qzeros, self.bits, self.groupsize, self.infeatures, self.outfeatures)
        out = out + self.bias if self.bias is not None else out
        return out
