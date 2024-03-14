import torch
import math
import os


def lcm(a: int, b: int):
    return int(a * b / math.gcd(a, b))


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
    wf = torch.arange(0, bits).to(pack_tensor.device).view(1, 1, -1)
    out = torch.bitwise_right_shift(ori_int_tensor.unsqueeze(-1), wf)
    torch.bitwise_and(out, 1, out=out)
    out = out.reshape(ori_int_tensor.shape[0], -1, 32)
    wf1 = torch.arange(0, 32, 1).to(pack_tensor.device).view(1, 1, -1)
    out = torch.bitwise_left_shift(out, wf1)
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


def general_unpack_on_row(pack_tensor, ori_int32_tensor, bits: int):
    assert pack_tensor.shape[0] == ori_int32_tensor.shape[0] or pack_tensor.shape[1] == ori_int32_tensor.shape[1], ''
    ori_int32_tensor.mul_(0)
    if bits in [2, 4, 8]:
        return unpack_on_row_fast_248bit(pack_tensor, ori_int32_tensor, bits)
    return unpack_on_row_fast_any_bit(pack_tensor, ori_int32_tensor, bits)


class CompressWeight(object):
    def _quant_weight(self, weight, scales, zeros, g_idx, need_transpose=True):
        scale_zeros = zeros * scales
        scale_mat = scales[g_idx]
        scale_zeros_mat = scale_zeros[g_idx]
        intweight_T = torch.round((weight + scale_zeros_mat) / scale_mat).to(torch.int)
        return intweight_T

    def _dequant_weight(self, intweight, scales, zeros, g_idx):
        scale_zeros = zeros * scales
        scale_mat = scales[g_idx]
        scale_zeros_mat = scale_zeros[g_idx]
        qdq_weight_T = intweight * scale_mat - scale_zeros_mat.half()

        return qdq_weight_T

    def weight_qdq(self, linear, scales, zeros, g_idx=None):
        if linear.bias is not None:
            self.bias = linear.bias.clone().half()
        g_idx = self.g_idx.to(scales.device) if g_idx is None else g_idx
        weight = linear.weight.data
        q_weight = self._quant_weight(weight.T, scales.T, zeros.T, g_idx).T
        return self._dequant_weight(q_weight.T, scales.T, zeros.T, g_idx).T

    def unpack(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        qzeros = self.qzeros.to(device)
        qweight = self.qweight.to(device)
        scales = self.scales.to(device)

        weight_dim0 = self.infeatures
        if "GEMM" in self._get_name():
            qweight = qweight.T.contiguous()
            weight_dim0 = self.outfeatures

        weight = torch.zeros((weight_dim0, qweight.shape[1]), dtype=torch.int32, device=qweight.device)
        zeros = torch.zeros((self.infeatures // self.groupsize, self.outfeatures), dtype=torch.int32, device=qweight.device)
        general_unpack_on_row(qweight, weight, self.bits)
        general_unpack_on_row(qzeros, zeros, self.bits)

        if "GEMM" in self._get_name():
            zeros = zeros.T.contiguous()
        zeros = self.reverse_reorder_int_tensor(zeros)
        weight = self.reverse_reorder_int_tensor(weight)

        fp16_weight = self._dequant_weight(weight, scales, zeros, self.g_idx.to(device)).T
        # free memory
        weight = weight.to("cpu", non_blocking=True)
        # weight = (scales * (weight - zeros))
        # weight = weight.reshape(weight.shape[0] * weight.shape[1], weight.shape[2])
        fp16_weight = fp16_weight.to("cpu", non_blocking=True)
        zeros = zeros.to("cpu", non_blocking=True)
        scales = scales.to("cpu", non_blocking=True)
        return (fp16_weight, scales, zeros)

    def reorder_int_tensor(self, int_tensor):
        return int_tensor

    def reverse_reorder_int_tensor(self, int_tensor):
        return int_tensor

    def pack_qzeros_odd(self, intzeros, device):
        # why -1?
        # zeros_cuda = (zeros - 1).to(device).int()
        COMPATIBLE_WITH_AUTOGPTQ = int(os.environ.get("COMPATIBLE_WITH_AUTOGPTQ", "0"))
        zeros_cuda = (intzeros - COMPATIBLE_WITH_AUTOGPTQ)
        max_num_in_bits = 2**self.bits - 1
        zeros_cuda = (zeros_cuda.byte() & max_num_in_bits).int()
        qzeros_cuda = torch.zeros(
            (intzeros.shape[0], (intzeros.shape[1] * self.bits + 31) // 32), dtype=torch.int32, device=device)

        qzeros_cuda = qzeros_cuda.T
        zeros_cuda = zeros_cuda.T
        general_pack_on_row(qzeros_cuda, zeros_cuda, self.bits)

        self.qzeros = qzeros_cuda.T.contiguous().to("cpu", non_blocking=True)

    # odd bits, 3,5,6,7
    def pack_on_device_for_odd_bits(self, intweight_gpu, intzeros):
        device = intweight_gpu.device
        qweight_gpu = torch.zeros(
            ((intweight_gpu.shape[0] * self.bits + 31) // 32, intweight_gpu.shape[1]), dtype=torch.int32, device=device)

        general_pack_on_row(qweight_gpu, intweight_gpu, self.bits)
        self.qweight = qweight_gpu.to("cpu", non_blocking=True)

        self.pack_qzeros_odd(intzeros, device)

        if self.orig_fp_weight is not None:
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
        qzeros_cuda = torch.zeros((intzeros.shape[0], intzeros.shape[1]
                                  // 32 * self.bits), dtype=torch.int32, device=device)
        qzeros_cuda = qzeros_cuda.T.contiguous()
        zeros_cuda = zeros_cuda.T.contiguous()
        pack_on_row_fast_248bit(qzeros_cuda, zeros_cuda, self.bits)
        self.qzeros = qzeros_cuda.T.contiguous().to("cpu", non_blocking=True)

    def pack_on_device_for_even_bits(self, intweight_gpu, qzeros):
        device = intweight_gpu.device
        intweight_gpu = self.reorder_int_tensor(intweight_gpu)
        qzeros = self.reorder_int_tensor(qzeros)
        if "GEMM" in self._get_name():
            qzeros = qzeros.T.contiguous()
        assert intweight_gpu.shape[0] // 32 * self.bits == int(round(intweight_gpu.shape[0] * self.bits / 32 + 0.5))

        qweight_gpu = torch.zeros(
            (intweight_gpu.shape[0] // 32 * self.bits, intweight_gpu.shape[1]), dtype=torch.int32, device=device)

        pack_on_row_fast_248bit(qweight_gpu, intweight_gpu, self.bits)
        if "GEMM" in self._get_name():
            qweight_gpu = qweight_gpu.T.contiguous()
        self.qweight = qweight_gpu.to("cpu", non_blocking=True)

        assert max(1, qzeros.shape[1] // 32 * self.bits) == int(round(qzeros.shape[1] * self.bits / 32 + 0.5))
        self.pack_qzeros_even(qzeros, device)

        if self.orig_fp_weight is not None:
            fw, _, iz = self.unpack()
            assert (fw == self.orig_fp_weight.to(device)).all()

    def accelerate_pack_on_device(self, layer_weight, scales, zeros, g_idx=None, device="cuda"):
        self.scales = scales.T.contiguous().half().to("cpu", non_blocking=True)
        if g_idx is None:
            g_idx = self.g_idx.to(device) if g_idx is None else g_idx
        else:
            self.g_idx = g_idx.clone().to("cpu", non_blocking=True)

        intweight_gpu = self._quant_weight(layer_weight.T, scales.T, zeros.T, g_idx)

        qzeros = zeros.T.contiguous()

        if self.bits in [2, 4, 8]:
            return self.pack_on_device_for_even_bits(intweight_gpu, qzeros)
        else:
            return self.pack_on_device_for_odd_bits(intweight_gpu, qzeros)

    def pack(self, linear, scales, zeros, g_idx=None):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        scales = scales.to(device)
        zeros = zeros.to(device)
        layer_weight = linear.weight.data.to(device)
        return self.accelerate_pack_on_device(layer_weight, scales, zeros, g_idx, device)
