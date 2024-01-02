import torch
import math
import os

def lcm(a:int, b:int): 
    return int(a*b/math.gcd(a, b))

def pack_on_row_fast_248bit(pack_tensor, ori_int_tensor, bits):
    if pack_tensor.shape[0] == ori_int_tensor.shape[0]:
        ori_int_tensor = ori_int_tensor.T
        pack_tensor = pack_tensor.T
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
        
def pack_on_row_fast_anybit(pack_tensor, ori_int_tensor, bits):
    need_transpose = False
    if pack_tensor.shape[0] != ori_int_tensor.shape[0]:
        need_transpose = True
        ori_int_tensor = ori_int_tensor.T
    pack_tensor.mul_(0)
    wf = torch.arange(0, bits).to(pack_tensor.device).view(1,1, -1)
    out = torch.bitwise_right_shift(ori_int_tensor.unsqueeze(-1), wf)
    torch.bitwise_and(out, 1, out=out)
    out = out.reshape(ori_int_tensor.shape[0], -1, 32)
    wf1 = torch.arange(0, 32, 1).to(pack_tensor.device).view(1, 1, -1)
    out = torch.bitwise_left_shift(out,wf1)
    out = out.sum(dim=-1).int()

    if need_transpose:
        out = out.T.contiguous()
    pack_tensor.copy_(out)

        
def general_pack_on_row(pack_tensor, ori_int32_tensor, bits):
    assert pack_tensor.shape[0] == ori_int32_tensor.shape[0] or pack_tensor.shape[1] == ori_int32_tensor.shape[1], ''
    pack_tensor.mul_(0)
    if bits in [2, 4, 8]:
        return pack_on_row_fast_248bit(pack_tensor, ori_int32_tensor, bits)
    return pack_on_row_fast_anybit(pack_tensor, ori_int32_tensor, bits)

def unpack_on_row_fast_248bit(pack_tensor, ori_int_tensor, bits):
    need_transpose = False
    if pack_tensor.shape[0] != ori_int_tensor.shape[0]:
        need_transpose = True
        pack_tensor = pack_tensor.T
    wf = torch.arange(0, 32, bits).to(pack_tensor.device).unsqueeze(0)
    out = torch.bitwise_right_shift(torch.unsqueeze(pack_tensor, 2), wf.unsqueeze(0))
    out = out.reshape(pack_tensor.shape[0], -1)
    torch.bitwise_and(out, (2 ** bits) - 1, out=out).int()
    if need_transpose:
        out = out.T.contiguous()

    ori_int_tensor.copy_(out)

#@torch.jit.script
def unpack_on_row_fast_any_bit(pack_tensor: torch.Tensor, ori_int_tensor: torch.Tensor, bits: int):
    need_transpose = False
    if pack_tensor.shape[0] != ori_int_tensor.shape[0]:
        need_transpose = True
        pack_tensor = pack_tensor.T
    wf = torch.arange(0, 32, 1).to(pack_tensor.device).unsqueeze(0).unsqueeze(0)
    out = torch.bitwise_right_shift(torch.unsqueeze(pack_tensor, 2), wf).char()
    torch.bitwise_and(out, 1, out=out)

    out = out.reshape(pack_tensor.shape[0], -1, bits)
    wf1 = torch.arange(0, bits, 1).to(pack_tensor.device).unsqueeze(0).unsqueeze(0)

    out = torch.bitwise_left_shift(out, wf1).sum(dim=-1)
    if need_transpose:
        out = out.T.contiguous()
    ori_int_tensor.copy_(out)

#@torch.jit.script
def general_unpack_on_row(pack_tensor, ori_int32_tensor, bits:int):
    assert pack_tensor.shape[0] == ori_int32_tensor.shape[0] or pack_tensor.shape[1] == ori_int32_tensor.shape[1], ''
    ori_int32_tensor.mul_(0)
    if bits in [2, 4, 8]:
        return unpack_on_row_fast_248bit(pack_tensor, ori_int32_tensor, bits)
    return unpack_on_row_fast_any_bit(pack_tensor, ori_int32_tensor, bits)


class CompressWeight(object):
    def quant_weight(self, weight, scales, zeros, g_idx=None, need_transpose=True):
        device = weight.device
        scales = scales.t().contiguous().to(device)
        zeros = zeros.t().contiguous().to(device)
        g_idx = self.g_idx.long().to(device)
        scale_zeros = zeros * scales
        self.scales = (scales.clone().half() if self.scales.sum() == 0 else self.scales).cpu()

        # intweight = []
        # for idx in range(self.infeatures):
        #     intweight.append(torch.round((linear.weight.data[:, idx].to(device) + scale_zeros[self.g_idx[idx]].to(device)) / self.scales[self.g_idx[idx]].to(device)).to(torch.int)[:, None])
        # intweight = torch.cat(intweight, dim=1)

        scale_mat = scales[g_idx]
        scale_zeros_mat = scale_zeros[g_idx]
        intweight_T = torch.round((weight.T + scale_zeros_mat) / scale_mat).to(torch.int)

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
        device = intweight.device
        scales = self.scales.to(device)
        zeros = zeros.t().contiguous()
        scale_zeros = zeros * scales
        g_idx = self.g_idx.long().to(device)

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
        weight_dim0 = self.infeatures
        if "GEMM" in self._get_name():
            qweight = qweight.T.contiguous()
            weight_dim0 = self.outfeatures
        scales = self.scales
        scales = scales.reshape(-1, 1, scales.shape[-1])
        
        weight = torch.zeros((weight_dim0, qweight.shape[1]), dtype=torch.int32, device=qweight.device)
        zeros = torch.zeros((self.infeatures // self.groupsize, self.outfeatures), dtype=torch.int32, device=qweight.device)
        general_unpack_on_row(qweight, weight, self.bits)
        general_unpack_on_row(qzeros, zeros, self.bits)

        if "GEMM" in self._get_name():
            zeros = zeros.T.contiguous()
        zeros = self.reverse_reorder_int_tensor(zeros)
        weight = self.reverse_reorder_int_tensor(weight)

        fp16_weight = self.dequant_weight(weight.T, zeros.T)
        weight = weight.cpu()
        # weight = (scales * (weight - zeros))
        # weight = weight.reshape(weight.shape[0] * weight.shape[1], weight.shape[2])
        return fp16_weight, self.scales, zeros.cpu()


    def reorder_int_tensor(self, int_tensor):
        return int_tensor

    def reverse_reorder_int_tensor(self, int_tensor):
        return int_tensor

    def pack_qzeros_odd(self, intzeros, device):
        # why -1?
        # zeros_cuda = (zeros - 1).to(device).int()
        compatible_with_autogptq = int(os.environ.get("compatible_with_autogptq", "0"))
        zeros_cuda = (intzeros - compatible_with_autogptq)
        max_num_in_bits = 2**self.bits - 1
        zeros_cuda = (zeros_cuda.byte() & max_num_in_bits).int()
        qzeros_cuda = torch.zeros(
            (intzeros.shape[0], (intzeros.shape[1] * self.bits + 31) // 32), dtype=torch.int32, device=device)

        qzeros_cuda = qzeros_cuda.T.contiguous()
        zeros_cuda = zeros_cuda.T.contiguous()
        general_pack_on_row(qzeros_cuda, zeros_cuda, self.bits)

        self.qzeros = qzeros_cuda.T.contiguous().cpu()

    # odd bits, 3,5,6,7
    def pack_on_device_for_odd_bits(self, intweight_gpu, intzeros):
        device = intweight_gpu.device
        import time
        s = time.time()
        qweight_gpu = torch.zeros(
            ((intweight_gpu.shape[0] * self.bits+31) // 32, intweight_gpu.shape[1]), dtype=torch.int32, device=device)

        general_pack_on_row(qweight_gpu, intweight_gpu, self.bits)
        self.qweight = qweight_gpu.cpu()

        self.pack_qzeros_odd(intzeros, device)

        if self.orig_fp_weight != None:
            fw, _, iz = self.unpack()
            assert (fw == self.orig_fp_weight.to(device)).all()

    def pack_qzeros_even(self, intzeros, device):
        intzeros = intzeros.int()
        # why -1?
        # zeros_cuda = (zeros - 1).to(device).int()
        COMPATIBLE_WITH_AUTOGPTQ = int(os.environ.get("COMPATIBLE_WITH_AUTOGPTQ", "0"))
        zeros_cuda = (intzeros - COMPATIBLE_WITH_AUTOGPTQ)
        max_num_in_bits = 2**self.bits - 1
        zeros_cuda = (zeros_cuda.byte() & max_num_in_bits).int()
        qzeros_cuda = torch.zeros((intzeros.shape[0], intzeros.shape[1] //
                                  32 * self.bits), dtype=torch.int32, device=device)
        i = 0
        col = 0
        qzeros_cuda = qzeros_cuda.T.contiguous()
        zeros_cuda = zeros_cuda.T.contiguous()
        pack_on_row_fast_248bit(qzeros_cuda, zeros_cuda, self.bits)
        self.qzeros = qzeros_cuda.T.contiguous().cpu()

    def pack_on_device_for_even_bits(self, intweight_gpu, qzeros):
        device = intweight_gpu.device
        compress_ratio = (32 // self.bits)
        intweight_gpu = self.reorder_int_tensor(intweight_gpu)
        qzeros = self.reorder_int_tensor(qzeros)
        if "GEMM" in self._get_name():
            qzeros = qzeros.T.contiguous()
        assert intweight_gpu.shape[0] // 32 * self.bits == int(round(intweight_gpu.shape[0] * self.bits / 32 + 0.5))
        import time
        s = time.time()
        qweight_gpu = torch.zeros(
            (intweight_gpu.shape[0] // 32 * self.bits, intweight_gpu.shape[1]), dtype=torch.int32, device=device)

        
        pack_on_row_fast_248bit(qweight_gpu, intweight_gpu, self.bits)
        e1 = time.time()-s
        if "GEMM" in self._get_name():
            qweight_gpu = qweight_gpu.T.contiguous()
        self.qweight = qweight_gpu.cpu()

        assert qzeros.shape[1] // 32 * self.bits == int(round(qzeros.shape[1] * self.bits / 32 + 0.5))
        self.pack_qzeros_even(qzeros, device)

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

        qzeros = zeros.t().contiguous()

        if self.bits in [2, 4, 8]:
            return self.pack_on_device_for_even_bits(intweight_gpu, qzeros)
        else:
            return self.pack_on_device_for_odd_bits(intweight_gpu, qzeros)

    def pack(self, linear, scales, zeros, g_idx=None):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return self.accelerate_pack_on_device(linear, scales, zeros, g_idx, device)
