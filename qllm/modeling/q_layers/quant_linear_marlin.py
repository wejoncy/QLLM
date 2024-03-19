import numpy as np
import torch
import torch.nn as nn
from .compress_weight import CompressWeight
from .ext_package_checker import has_awq_inference_engine
from ...utils import comm_utils

if has_awq_inference_engine():
    from qllm import awq_inference_engine as marlin_cuda
else:
    print("marlin_cuda is not installed. marlin_cuda is not use")

DEBUG_ = False

# Precompute permutations for Marlin weight and scale shuffling


def _get_perms():
    perm = []
    for i in range(32):
        perm1 = []
        col = i // 4
        for block in [0, 1]:
            for row in [2 * (i % 4), 2 * (i % 4) + 1, 2 * (i % 4 + 4), 2 * (i % 4 + 4) + 1]:
                perm1.append(16 * row + col + 8 * block)
        for j in range(4):
            perm.extend([p + 256 * j for p in perm1])

    perm = np.array(perm)
    interleave = np.array([0, 2, 4, 6, 1, 3, 5, 7])
    perm = perm.reshape((-1, 8))[:, interleave].ravel()
    perm = torch.from_numpy(perm)
    scale_perm = []
    for i in range(8):
        scale_perm.extend([i + 8 * j for j in range(8)])
    scale_perm_single = []
    for i in range(4):
        scale_perm_single.extend([2 * i + j for j in [0, 1, 8, 9, 16, 17, 24, 25]])
    return perm, scale_perm, scale_perm_single


_perm, _scale_perm, _scale_perm_single = _get_perms()


def mul(A, B, C, s, workspace, thread_k=-1, thread_n=-1, sms=-1, max_par=16):
    """Marlin FP16xINT4 multiply; can be used within `torch.compile`.
    @A: `torch.half` input matrix of shape `(m, k)` in standard row-major layout
    @B: `torch.int` weight matrix of original shape `(k, n)` in Marlin format; see `Layer.pack()`
    @C: `torch.half` out matrix of shape `(m, n)` in standard row-major layout
    @s: `torch.half` scales of shape `(m / group_size, n)`
    @workspace: `torch.int` tensor with at least as many entries as there a GPU SMs (256 is usually safe)
    @thread_k: `k` size of a thread_tile in `B` (can usually be left as auto -1)
    @thread_n: `n` size of a thread_tile in `B` (can usually be left as auto -1)
    @sms: number of SMs to use for the kernel (can usually be left as auto -1)
    @max_par: maximum number of batch 64 problems to solve in parallel for large input sizes
    """
    marlin_cuda.mul(A, B, C, s, workspace, thread_k, thread_n, sms, max_par)


class QuantLinearMarlin(nn.Module, CompressWeight):
    def __init__(self, bits, group_size, infeatures, outfeatures, bias):
        super().__init__()
        if bits not in [2, 3, 4, 5, 6, 7, 8]:
            raise NotImplementedError("Only 2,4,5,6,7,8 bits are supported.")
        self.infeatures = infeatures
        self.outfeatures = outfeatures
        self.bits = bits
        self.orig_fp_weight = None
        self.maxq = 2**self.bits - 1
        self.group_size = group_size if group_size != -1 else infeatures
        self.act_order = None
        self.pack_mode = "MARLIN"
        if infeatures % 128 != 0 or outfeatures != 256 == 0:
            raise ValueError("`infeatures` must be divisible by 128 and `outfeatures` by 256.")
        if bits not in [4]:
            raise NotImplementedError("Only 4 bits are supported.")
        if group_size not in [-1, 128] and group_size != infeatures:
            raise ValueError("Only group_size -1 and 128 are supported.")
        if infeatures % group_size != 0:
            raise ValueError("`infeatures` must be divisible by `group_size`.")
        self.register_buffer(
            "qweight",
            torch.zeros((self.infeatures // 16, self.outfeatures * 16 // 8), dtype=torch.int),
        )
        self.register_buffer(
            "scales", torch.zeros((self.infeatures // group_size, self.outfeatures), dtype=torch.float16)
        )
        self.register_buffer("workspace", torch.zeros(self.outfeatures // 128 * 16, dtype=torch.int), persistent=False)
        self.g_idx = None
        if bias:
            self.register_buffer("bias", torch.zeros((outfeatures), dtype=torch.float16))
        else:
            self.bias = None

    def pack(self, linear, scales, zeros, g_idx=None):
        if g_idx is not None:
            self.act_order = g_idx[: self.group_size // self.bits].sum().item() != 0
            assert self.act_order is False
        assert zeros is None or torch.all(zeros == 8)
        if linear.weight.dtype != torch.half:
            raise ValueError("Only `torch.half` weights are supported.")
        device = torch.device("cuda")
        tile = 16
        maxq = 2**4 - 1
        s = scales.to(device).t()
        w = linear.weight.data.to(device).t()
        if self.group_size != self.infeatures:
            w = w.reshape((-1, self.group_size, self.outfeatures))
            w = w.permute(1, 0, 2)
            w = w.reshape((self.group_size, -1))
            s = s.reshape((1, -1))
        w = torch.round(w / s).int()
        w += (maxq + 1) // 2
        w = torch.clamp(w, 0, maxq)
        if self.group_size != self.infeatures:
            w = w.reshape((self.group_size, -1, self.outfeatures))
            w = w.permute(1, 0, 2)
            w = w.reshape((self.infeatures, self.outfeatures)).contiguous()
            s = s.reshape((-1, len(_scale_perm)))[:, _scale_perm]
        else:
            s = s.reshape((-1, len(_scale_perm_single)))[:, _scale_perm_single]
        s = s.reshape((-1, self.outfeatures)).contiguous()
        w = w.reshape((self.infeatures // tile, tile, self.outfeatures // tile, tile))
        w = w.permute((0, 2, 1, 3))
        w = w.reshape((self.infeatures // tile, self.outfeatures * tile))
        res = w
        res = res.reshape((-1, _perm.numel()))[:, _perm].reshape(res.shape)
        q = torch.zeros((res.shape[0], res.shape[1] // 8), dtype=torch.int32, device=device)

        for i in range(8):
            q |= res[:, i::8] << (4 * i)
        q = q.to(torch.int32)
        self.qweight[:, :] = q.to(self.qweight.device)
        self.scales[:, :] = s.to(self.scales.device)
        if self.bias is not None:
            self.bias[:] = linear.bias.data.to(self.bias.device)

    def unpack(self):
        raise NotImplementedError("Marlin unpacking is not supported.")

    def forward(self, x):
        C = torch.empty(x.shape[:-1] + (self.scales.shape[1],), dtype=x.dtype, device=x.device)
        mul(x.view((-1, x.shape[-1])), self.qweight, C.view((-1, C.shape[-1])), self.scales, self.workspace)
        C = C + self.bias if self.bias is not None else C
        return C
