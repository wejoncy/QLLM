import torch
import math

import numpy as np

def lcm(a, b): 
    return int(a*b/math.gcd(a, b))



def pack_on_row_fast_248bit(pack_tensor, ori_int_tensor, bits):
    compress_ratio = (32 // bits)
    i = 0
    row = 0
    while row < pack_tensor.shape[0]:
        if bits in [2, 4, 8]:
            for j in range(i, i + compress_ratio):
                pack_tensor[row:] |= ori_int_tensor[j::compress_ratio] << (
                    bits * (j - i))
            break
        else:
            raise NotImplementedError("Only 2,4,8 bits are supported.")

def general_pack_on_row(pack_tensor, ori_int32_tensor, bits):
    pack_tensor.mul_(0)
    if bits in [2, 4, 8]:
        return pack_on_row_fast_248bit(pack_tensor, ori_int32_tensor, bits)
    row = 0
    fp16_row = 0
    last_bits = 0
    last_row = 0
    rounds = lcm(bits, 32)//32  # int32
    while row < pack_tensor.shape[0]:
        last_bits = 0
        last_row = 0
        for round_i in range(rounds):
            pack_tensor[row] |= last_row
            nums_of_rows = min((32-last_bits)//bits, ori_int32_tensor.shape[0]-fp16_row)
            for j in range(nums_of_rows):
                pack_tensor[row] |= ori_int32_tensor[fp16_row+j] << (last_bits+bits * j)
            fp16_row += nums_of_rows
            if fp16_row >= ori_int32_tensor.shape[0]:
                row += 1
                assert row == pack_tensor.shape[0]
                break
            rest_bits = 32-last_bits-nums_of_rows*bits
            if rest_bits > 0:
                last_bits = bits - rest_bits
                last_row = ori_int32_tensor[fp16_row]
                pack_tensor[row] |= (last_row & ((1 << rest_bits)-1)) << (32-rest_bits)
                last_row = last_row >> rest_bits
            else:
                last_row = 0
                last_bits = 0
                fp16_row -= 1

            row += 1
            fp16_row += 1

def unpack_on_row_fast_248bit(pack_tensor, ori_int_tensor, bits):
    wf = torch.arange(0, 32, bits).to(pack_tensor.device).unsqueeze(0)
    out = torch.bitwise_right_shift(torch.unsqueeze(pack_tensor, 2), wf.unsqueeze(0))
    out = out.view(ori_int_tensor.shape)
    torch.bitwise_and(out, (2 ** bits) - 1, out=ori_int_tensor)

def general_unpack_on_row(pack_tensor, ori_int32_tensor, bits):
    ori_int32_tensor.mul_(0)
    if bits in [2, 4, 8]:
        return unpack_on_row_fast_248bit(pack_tensor, ori_int32_tensor, bits)
    row = 0
    fp16_row = 0
    last_bits = 0
    last_row = 0
    def lcm(a, b): return int(a*b/math.gcd(a, b))
    rounds = int(lcm(bits, 32)//32)  # int32
    while row < pack_tensor.shape[0]:
        last_bits = 0
        last_row = 0
        for round_i in range(rounds):
            ori_int32_tensor[last_row] |= (pack_tensor[row] & ((1 << last_bits)-1)) << (bits-last_bits)
            nums_of_rows = min((32-last_bits)//bits, ori_int32_tensor.shape[0]-fp16_row)
            for j in range(nums_of_rows):
                ori_int32_tensor[fp16_row] |= (pack_tensor[row] >> (last_bits+bits * j)) & ((1 << bits)-1)
                fp16_row += 1
            if fp16_row >= ori_int32_tensor.shape[0]:
                row += 1
                assert row == pack_tensor.shape[0]
                break
            rest_bits = 32-last_bits-nums_of_rows*bits
            if rest_bits > 0:
                last_bits = bits - rest_bits
                last_row = fp16_row
                ori_int32_tensor[fp16_row] |= (pack_tensor[row] >> (32-rest_bits)) & ((1 << rest_bits)-1)
            else:
                last_row = 0
                last_bits = 0
                fp16_row -= 1

            row += 1
            fp16_row += 1


class CompressWeight(object):
    def quant_weight(self, weight, scales, zeros, g_idx=None, need_transpose=True):
        device = weight.device
        scales = scales.t().contiguous()
        zeros = zeros.t().contiguous()
        g_idx = self.g_idx.long().to(device)
        scale_zeros = zeros * scales
        self.scales = (scales.clone().half() if self.scales.sum() == 0 else self.scales).cpu()

        # intweight = []
        # for idx in range(self.infeatures):
        #     intweight.append(torch.round((linear.weight.data[:, idx].to(device) + scale_zeros[self.g_idx[idx]].to(device)) / self.scales[self.g_idx[idx]].to(device)).to(torch.int)[:, None])
        # intweight = torch.cat(intweight, dim=1)

        scale_mat = scales[g_idx]
        scale_zeros_mat = scale_zeros[g_idx]
        intweight_T = torch.round((weight.T+scale_zeros_mat)/scale_mat).to(torch.int)

        # when shouldn't use scale_zeros_mat
        # zeros=zeros.to(device)
        # zeros_mat = zeros[self.g_idx.long().to(device)]
        # intweight_T  = torch.round((linear.weight.to(device).T/scale_mat)+zeros_mat).to(torch.int)

        # assert (intweight_T.T == intweight).all()
        if not need_transpose:
            return intweight_T.cpu()
        return intweight_T.T.cpu()

    def dequant_weight(self, intweight, zeros):
        # scales = scales.t().contiguous()
        scales = self.scales
        zeros = zeros.t().contiguous()
        scale_zeros = zeros * scales
        g_idx = self.g_idx.long()

        # qdq_weight=linear.weight.clone().to(device)
        # for idx in range(self.infeatures):
        #     qdq_weight[:, idx] = intweight[:,idx].to(device)*self.scales[self.g_idx[idx]].to(device) - scale_zeros[self.g_idx[idx]].to(device).half()

        scale_mat = scales[g_idx]
        scale_zeros_mat = scale_zeros[g_idx].half()
        qdq_weight_T = intweight.T*scale_mat-scale_zeros_mat.half()

        # when shouldn't use scale_zeros_mat
        # zeros=zeros.to(device)
        # zeros_mat=zeros[self.g_idx.long().to(device)]
        # qdq_weight_T = (intweight.to(device).T-zeros_mat)*scale_mat

        # assert (qdq_weight_T.T == qdq_weight).all()
        return qdq_weight_T.T.cpu()

    def weight_qdq(self, linear, scales, zeros, g_idx=None):
        self.g_idx = g_idx.clone() if g_idx is not None else self.g_idx
        if linear.bias is not None:
            self.bias = linear.bias.clone().half()
        q_weight = self.quant_weight(linear.weight.data, scales, zeros, self.g_idx)
        return self.dequant_weight(q_weight, zeros)

    def unpack(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        qzeros = self.qzeros.to(device)
        qweight = self.qweight.to(device)
        if "GEMM" in self._get_name():
            qweight = qweight.T.contiguous()
        scales = self.scales
        scales = scales.reshape(-1, 1, scales.shape[-1])
        import os
        load_from_autogptq = int(os.environ.get('load_from_autogptq', "0"))

        if self.bits in [2, 4, 8]:
            wf = torch.tensor(list(range(0, 32, self.bits)), dtype=torch.int32, device=qzeros.device).unsqueeze(0)
            zeros = torch.bitwise_right_shift(torch.unsqueeze(qzeros, 2), wf.unsqueeze(0)).to(
                torch.int16 if self.bits == 8 else torch.int8)
            torch.bitwise_and(zeros, (2 ** self.bits) - 1, out=zeros)

            zeros = zeros + load_from_autogptq
            zeros = zeros.reshape(-1, 1, zeros.shape[1] * zeros.shape[2])

            weight = torch.bitwise_right_shift(torch.unsqueeze(
                qweight, 1), wf.unsqueeze(-1)).to(torch.int16 if self.bits == 8 else torch.int8)
            torch.bitwise_and(weight, (2 ** self.bits) - 1, out=weight)
            weight = weight.reshape(-1, self.groupsize, weight.shape[2])

            weight = weight.view(-1, weight.shape[-1])
            zeros = zeros.view(-1, zeros.shape[-1])
        else:
            weight = torch.zeros((self.infeatures, qweight.shape[1]), dtype=torch.int8, device=qweight.device)
            general_unpack_on_row(qweight, weight, self.bits)
            zeros = torch.zeros((self.infeatures//self.groupsize,
                                qweight.shape[1]), dtype=torch.int8, device=qweight.device)
            zeros = zeros.T
            general_unpack_on_row(qzeros.T, zeros, self.bits)
            zeros = zeros.T
            zeros = zeros + load_from_autogptq

        if "GEMM" in self._get_name():
            zeros = zeros.T.contiguous()
        zeros = self.reverse_reorder_int_tensor(zeros)
        weight = self.reverse_reorder_int_tensor(weight)

        fp16_weight = self.dequant_weight(weight.T, zeros.T).to(device)
        # weight = (scales * (weight - zeros))
        # weight = weight.reshape(weight.shape[0] * weight.shape[1], weight.shape[2])
        return fp16_weight, self.scales, zeros

    # odd bits, 3,5,6,7
    def pack_on_device_for_odd_bits(self, intweight_gpu, intzeros):
        import time
        s = time.time()
        qweight_gpu = torch.zeros(
            ((intweight_gpu.shape[0] * self.bits+31) // 32, intweight_gpu.shape[1]), dtype=torch.int32, device=intweight_gpu.device)

        general_pack_on_row(qweight_gpu, intweight_gpu, self.bits)
        self.qweight = qweight_gpu.cpu()

        # why -1?
        # zeros_cuda = (zeros - 1).to(device).int()
        zeros_cuda = (intzeros).int()
        qzeros_cuda = torch.zeros(
            (intzeros.shape[0], (intzeros.shape[1] * self.bits+31) // 32), dtype=torch.int32, device=zeros_cuda.device)

        qzeros_cuda = qzeros_cuda.T.contiguous()
        zeros_cuda = zeros_cuda.T.contiguous()
        general_pack_on_row(qzeros_cuda, zeros_cuda, self.bits)

        self.qzeros = qzeros_cuda.T.contiguous().cpu()
        e2 = time.time()-s

        if self.orig_fp_weight != None:
            fw, _, iz = self.unpack()
            assert (fw == self.orig_fp_weight.to(device)).all()

    def reorder_int_tensor(self, int_tensor):
        return int_tensor

    def reverse_reorder_int_tensor(self, int_tensor):
        return int_tensor

    def pack_on_device_for_even_bits(self, intweight_gpu, intzeros):
        compress_ratio = (32 // self.bits)
        intweight_gpu = self.reorder_int_tensor(intweight_gpu)
        intzeros = self.reorder_int_tensor(intzeros)
        if "GEMM" in self._get_name():
            intzeros = intzeros.T.contiguous()
        assert intweight_gpu.shape[0] // 32 * self.bits == int(round(intweight_gpu.shape[0] * self.bits / 32 + 0.5))
        import time
        s = time.time()
        qweight_gpu = torch.zeros(
            (intweight_gpu.shape[0] // 32 * self.bits, intweight_gpu.shape[1]), dtype=torch.int32, device=intweight_gpu.device)

        
        pack_on_row_fast_248bit(qweight_gpu, intweight_gpu, self.bits)
        e1 = time.time()-s
        if "GEMM" in self._get_name():
            qweight_gpu = qweight_gpu.T.contiguous()
        self.qweight = qweight_gpu.cpu()

        assert intzeros.shape[1] // 32 * self.bits == int(round(intzeros.shape[1] * self.bits / 32 + 0.5))
        s = time.time()
        # why -1?
        # zeros_cuda = (zeros - 1).to(device).int()
        zeros_cuda = (intzeros).int()
        qzeros_cuda = torch.zeros((intzeros.shape[0], intzeros.shape[1] //
                                  32 * self.bits), dtype=torch.int32, device=zeros_cuda.device)
        i = 0
        col = 0
        qzeros_cuda = qzeros_cuda.T.contiguous()
        zeros_cuda = zeros_cuda.T.contiguous()
        pack_on_row_fast_248bit(qzeros_cuda, zeros_cuda, self.bits)
        self.qzeros = qzeros_cuda.T.contiguous().cpu()
        e2 = time.time()-s

        if self.orig_fp_weight != None:
            fw, _, iz = self.unpack()
            assert (fw == self.orig_fp_weight.to(device)).all()

    def accelerate_pack_on_device(self, linear, scales, zeros, g_idx=None, device="cuda"):
        scales = scales.to(device)
        zeros = zeros.to(device)
        layer_weight = linear.weight.data.to(device)

        self.g_idx = g_idx.clone() if g_idx is not None else self.g_idx
        intweight = self.quant_weight(layer_weight, scales, zeros, g_idx, need_transpose=False)
        intweight_gpu = intweight.to(device)

        intzeros = zeros.t().contiguous().int()

        if self.bits in [2, 4, 8]:
            return self.pack_on_device_for_even_bits(intweight_gpu, intzeros)
        else:
            return self.pack_on_device_for_odd_bits(intweight_gpu, intzeros)

    def pack(self, linear, scales, zeros, g_idx=None):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return self.accelerate_pack_on_device(linear, scales, zeros, g_idx, device)
