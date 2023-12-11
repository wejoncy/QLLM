import math
import os
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import custom_bwd, custom_fwd
from .compress_weight import CompressWeight


try:
    import triton
    import triton.language as tl
    from . import custom_autotune

    # code based https://github.com/fpgaminer/GPTQ-triton
    @custom_autotune.autotune(
        configs=[
            triton.Config({
                'BLOCK_SIZE_M': 64,
                'BLOCK_SIZE_N': 256,
                'BLOCK_SIZE_K': 32,
                'GROUP_SIZE_M': 8
            }, num_stages=4, num_warps=4),
            triton.Config({
                'BLOCK_SIZE_M': 128,
                'BLOCK_SIZE_N': 128,
                'BLOCK_SIZE_K': 32,
                'GROUP_SIZE_M': 8
            }, num_stages=4, num_warps=4),
            triton.Config({
                'BLOCK_SIZE_M': 64,
                'BLOCK_SIZE_N': 128,
                'BLOCK_SIZE_K': 32,
                'GROUP_SIZE_M': 8
            }, num_stages=4, num_warps=4),
            triton.Config({
                'BLOCK_SIZE_M': 128,
                'BLOCK_SIZE_N': 32,
                'BLOCK_SIZE_K': 32,
                'GROUP_SIZE_M': 8
            }, num_stages=4, num_warps=4),
            triton.Config({
                'BLOCK_SIZE_M': 64,
                'BLOCK_SIZE_N': 64,
                'BLOCK_SIZE_K': 32,
                'GROUP_SIZE_M': 8
            }, num_stages=4, num_warps=4),
            triton.Config({
                'BLOCK_SIZE_M': 64,
                'BLOCK_SIZE_N': 128,
                'BLOCK_SIZE_K': 32,
                'GROUP_SIZE_M': 8
            }, num_stages=2, num_warps=8),
            triton.Config({
                'BLOCK_SIZE_M': 64,
                'BLOCK_SIZE_N': 64,
                'BLOCK_SIZE_K': 64,
                'GROUP_SIZE_M': 8
            }, num_stages=3, num_warps=8),
            triton.Config({
                'BLOCK_SIZE_M': 32,
                'BLOCK_SIZE_N': 32,
                'BLOCK_SIZE_K': 128,
                'GROUP_SIZE_M': 8
            }, num_stages=2, num_warps=4),
        ],
        key=['M', 'N', 'K'],
        nearest_power_of_two=True,
        prune_configs_by={
            'early_config_prune': custom_autotune.matmul248_kernel_config_pruner,
            'perf_model': None,
            'top_k': None,
        },
    )
    @triton.jit
    def matmul_248_kernel(a_ptr, b_ptr, c_ptr, scales_ptr, zeros_ptr, g_ptr, M, N, K, bits, maxq, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, stride_scales, stride_zeros,
                          BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr):
        """
        Compute the matrix multiplication C = A x B.
        A is of shape (M, K) float16
        B is of shape (K//8, N) int32
        C is of shape (M, N) float16
        scales is of shape (G, N) float16
        zeros is of shape (G, N) float16
        g_ptr is of shape (K) int32 
        """
        infearure_per_bits = 32 // bits

        pid = tl.program_id(axis=0)
        num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
        num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
        num_pid_k = tl.cdiv(K, BLOCK_SIZE_K)
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + (pid % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

        offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        offs_k = tl.arange(0, BLOCK_SIZE_K)
        a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)  # (BLOCK_SIZE_M, BLOCK_SIZE_K)
        a_mask = (offs_am[:, None] < M)
        # b_ptrs is set up such that it repeats elements along the K axis 8 times
        b_ptrs = b_ptr + ((offs_k[:, None] // infearure_per_bits) * stride_bk +
                          offs_bn[None, :] * stride_bn)  # (BLOCK_SIZE_K, BLOCK_SIZE_N)
        g_ptrs = g_ptr + offs_k
        # shifter is used to extract the N bits of each element in the 32-bit word from B
        scales_ptrs = scales_ptr + offs_bn[None, :]
        zeros_ptrs = zeros_ptr + (offs_bn[None, :] // infearure_per_bits)

        shifter = (offs_k % infearure_per_bits) * bits
        zeros_shifter = (offs_bn % infearure_per_bits) * bits
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

        for k in range(0, num_pid_k):
            g_idx = tl.load(g_ptrs)

            # Fetch scales and zeros; these are per-outfeature and thus reused in the inner loop
            scales = tl.load(scales_ptrs + g_idx[:, None] * stride_scales)  # (BLOCK_SIZE_K, BLOCK_SIZE_N,)
            zeros = tl.load(zeros_ptrs + g_idx[:, None] * stride_zeros)  # (BLOCK_SIZE_K, BLOCK_SIZE_N,)

            zeros = (zeros >> zeros_shifter[None, :]) & maxq
            # zeros = (zeros + 1)

            a = tl.load(a_ptrs, mask=a_mask, other=0.)  # (BLOCK_SIZE_M, BLOCK_SIZE_K)
            b = tl.load(b_ptrs)  # (BLOCK_SIZE_K, BLOCK_SIZE_N), but repeated

            # Now we need to unpack b (which is N-bit values) into 32-bit values
            b = (b >> shifter[:, None]) & maxq  # Extract the N-bit values
            # b = (b - zeros) * scales  # Scale and shift

            b = b * scales - (scales*zeros).to(tl.float16)  # Scale and shift

            accumulator += tl.dot(a, b)
            a_ptrs += BLOCK_SIZE_K
            b_ptrs += (BLOCK_SIZE_K // infearure_per_bits) * stride_bk
            g_ptrs += BLOCK_SIZE_K

        c_ptrs = c_ptr + stride_cm * offs_am[:, None] + stride_cn * offs_bn[None, :]
        c_mask = (offs_am[:, None] < M) & (offs_bn[None, :] < N)
        tl.store(c_ptrs, accumulator.to(tl.float16), mask=c_mask)

    @custom_autotune.autotune(configs=[
        triton.Config({
            'BLOCK_SIZE_M': 64,
            'BLOCK_SIZE_N': 32,
            'BLOCK_SIZE_K': 256,
            'GROUP_SIZE_M': 8
        }, num_stages=4, num_warps=4),
        triton.Config({
            'BLOCK_SIZE_M': 128,
            'BLOCK_SIZE_N': 32,
            'BLOCK_SIZE_K': 128,
            'GROUP_SIZE_M': 8
        }, num_stages=4, num_warps=4),
        triton.Config({
            'BLOCK_SIZE_M': 64,
            'BLOCK_SIZE_N': 32,
            'BLOCK_SIZE_K': 128,
            'GROUP_SIZE_M': 8
        }, num_stages=4, num_warps=4),
        triton.Config({
            'BLOCK_SIZE_M': 128,
            'BLOCK_SIZE_N': 32,
            'BLOCK_SIZE_K': 32,
            'GROUP_SIZE_M': 8
        }, num_stages=4, num_warps=4),
        triton.Config({
            'BLOCK_SIZE_M': 64,
            'BLOCK_SIZE_N': 32,
            'BLOCK_SIZE_K': 64,
            'GROUP_SIZE_M': 8
        }, num_stages=4, num_warps=4),
        triton.Config({
            'BLOCK_SIZE_M': 64,
            'BLOCK_SIZE_N': 32,
            'BLOCK_SIZE_K': 128,
            'GROUP_SIZE_M': 8
        }, num_stages=2, num_warps=8),
        triton.Config({
            'BLOCK_SIZE_M': 64,
            'BLOCK_SIZE_N': 64,
            'BLOCK_SIZE_K': 64,
            'GROUP_SIZE_M': 8
        }, num_stages=3, num_warps=8),
        triton.Config({
            'BLOCK_SIZE_M': 32,
            'BLOCK_SIZE_N': 128,
            'BLOCK_SIZE_K': 32,
            'GROUP_SIZE_M': 8
        }, num_stages=2, num_warps=4),
    ],
        key=['M', 'N', 'K'],
        nearest_power_of_two=True)
    @triton.jit
    def transpose_matmul_248_kernel(a_ptr, b_ptr, c_ptr, scales_ptr, zeros_ptr, g_ptr, M, N, K, bits, maxq, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, stride_scales,
                                    stride_zeros, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr):
        """
        Compute the matrix multiplication C = A x B.
        A is of shape (M, N) float16
        B is of shape (K//8, N) int32
        C is of shape (M, K) float16
        scales is of shape (G, N) float16
        zeros is of shape (G, N) float16
        g_ptr is of shape (K) int32 
        """
        infearure_per_bits = 32 // bits

        pid = tl.program_id(axis=0)
        num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
        num_pid_k = tl.cdiv(K, BLOCK_SIZE_K)
        num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
        num_pid_in_group = GROUP_SIZE_M * num_pid_k
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + (pid % group_size_m)
        pid_k = (pid % num_pid_in_group) // group_size_m

        offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_bk = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
        offs_n = tl.arange(0, BLOCK_SIZE_N)
        a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_n[None, :] * stride_ak)  # (BLOCK_SIZE_M, BLOCK_SIZE_N)
        a_mask = (offs_am[:, None] < M)
        # b_ptrs is set up such that it repeats elements along the K axis 8 times
        b_ptrs = b_ptr + ((offs_bk[:, None] // infearure_per_bits) * stride_bk +
                          offs_n[None, :] * stride_bn)  # (BLOCK_SIZE_K, BLOCK_SIZE_N)
        g_ptrs = g_ptr + offs_bk
        g_idx = tl.load(g_ptrs)

        # shifter is used to extract the N bits of each element in the 32-bit word from B
        scales_ptrs = scales_ptr + offs_n[None, :] + g_idx[:, None] * stride_scales
        zeros_ptrs = zeros_ptr + (offs_n[None, :] // infearure_per_bits) + g_idx[:, None] * stride_zeros

        shifter = (offs_bk % infearure_per_bits) * bits
        zeros_shifter = (offs_n % infearure_per_bits) * bits
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)

        for n in range(0, num_pid_n):
            # Fetch scales and zeros; these are per-outfeature and thus reused in the inner loop
            scales = tl.load(scales_ptrs)  # (BLOCK_SIZE_K, BLOCK_SIZE_N,)
            zeros = tl.load(zeros_ptrs)  # (BLOCK_SIZE_K, BLOCK_SIZE_N,)

            zeros = (zeros >> zeros_shifter[None, :]) & maxq
            zeros = (zeros + 1)

            a = tl.load(a_ptrs, mask=a_mask, other=0.)  # (BLOCK_SIZE_M, BLOCK_SIZE_N)
            b = tl.load(b_ptrs)  # (BLOCK_SIZE_K, BLOCK_SIZE_N), but repeated

            # Now we need to unpack b (which is N-bit values) into 32-bit values
            b = (b >> shifter[:, None]) & maxq  # Extract the N-bit values
            b = (b - zeros) * scales  # Scale and shift
            b = tl.trans(b)

            accumulator += tl.dot(a, b)
            a_ptrs += BLOCK_SIZE_N
            b_ptrs += BLOCK_SIZE_N
            scales_ptrs += BLOCK_SIZE_N
            zeros_ptrs += (BLOCK_SIZE_N // infearure_per_bits)

        c_ptrs = c_ptr + stride_cm * offs_am[:, None] + stride_cn * offs_bk[None, :]
        c_mask = (offs_am[:, None] < M) & (offs_bk[None, :] < K)
        tl.store(c_ptrs, accumulator, mask=c_mask)
except Exception as e:
    print('triton not installed.')


def matmul248(input, qweight, scales, qzeros, g_idx, bits, maxq):
    os.environ['TOKENIZERS_PARALLELISM'] = "false"
    with torch.cuda.device(input.device):
        output = torch.empty((input.shape[0], qweight.shape[1]), device=input.device, dtype=torch.float16)
        def grid(META): return (triton.cdiv(input.shape[0], META['BLOCK_SIZE_M'])
                                * triton.cdiv(qweight.shape[1], META['BLOCK_SIZE_N']), )
        matmul_248_kernel[grid](input, qweight, output, scales, qzeros, g_idx, input.shape[0], qweight.shape[1], input.shape[1], bits, maxq, input.stride(0), input.stride(1), qweight.stride(0),
                                qweight.stride(1), output.stride(0), output.stride(1), scales.stride(0), qzeros.stride(0))
        return output


def transpose_matmul248(input, qweight, scales, qzeros, g_idx, bits, maxq):
    with torch.cuda.device(input.device):
        output_dim = (qweight.shape[0] * 32) // bits
        output = torch.empty((input.shape[0], output_dim), device=input.device, dtype=torch.float16)
        def grid(META): return (triton.cdiv(input.shape[0], META['BLOCK_SIZE_M'])
                                * triton.cdiv(output_dim, META['BLOCK_SIZE_K']), )
        transpose_matmul_248_kernel[grid](input, qweight, output, scales, qzeros, g_idx, input.shape[0], qweight.shape[1], output_dim, bits, maxq, input.stride(0), input.stride(1), qweight.stride(0),
                                          qweight.stride(1), output.stride(0), output.stride(1), scales.stride(0), qzeros.stride(0))
        return output

class QuantLinear(nn.Module, CompressWeight):
    def __init__(self, in_features, out_features, bias=True, bits=8, maxq=255, transpose=False):
        super().__init__()
        self.infeatures = in_features
        self.outfeatures = out_features
        self.bits = bits
        self.maxq = maxq
        self.transpose = transpose
        self.pack_mode = "GPTQ"

        self.qweight = nn.Parameter(torch.Tensor(out_features, in_features // 8))
        self.scales = nn.Parameter(torch.Tensor(out_features))
        self.qzeros = nn.Parameter(torch.Tensor(out_features))
        self.g_idx = nn.Parameter(torch.Tensor(out_features))
        self.register_buffer('bias', torch.zeros(out_features))
        self.reset_parameters()

    def forward(self, x):
        pass

class QuantLinearFunction(torch.autograd.Function):

    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(ctx, input, qweight, scales, qzeros, g_idx, bits, maxq):
        output = matmul248(input, qweight.contiguous(), scales, qzeros, g_idx, bits, maxq)
        # ctx.save_for_backward(qweight, scales, qzeros, g_idx)
        # ctx.bits, ctx.maxq = bits, maxq
        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        qweight, scales, qzeros, g_idx = ctx.saved_tensors
        bits, maxq = ctx.bits, ctx.maxq
        grad_input = None

        if ctx.needs_input_grad[0]:
            grad_input = transpose_matmul248(grad_output, qweight, scales, qzeros, g_idx, bits, maxq)
        return grad_input, None, None, None, None, None, None
        
def autotune_warmup_linear(model, transpose=False):
    """
    Pre-tunes the quantized kernel
    """
    from tqdm import tqdm

    kn_values = {}

    for _, m in model.named_modules():
        if not isinstance(m, QuantLinear):
            # if not isinstance(m, nn.Linear,loralib.MergedLinear, loralib.Linear):
            continue

        k = m.infeatures
        n = m.outfeatures

        if (k, n) not in kn_values:
            kn_values[(k, n)] = (m.qweight.cuda(), m.scales.cuda(), m.qzeros.cuda(), m.g_idx.cuda(), m.bits, m.maxq)

    print(f'Found {len(kn_values)} unique KN Linear values.')

    print('Warming up autotune cache ...')
    with torch.no_grad():
        for m in tqdm(range(0, 12)):
            m = 2**m  # [1, 2048]
            for (k, n), (qweight, scales, qzeros, g_idx, bits, maxq) in kn_values.items():
                a = torch.randn(m, k, dtype=torch.float16, device='cuda')
                matmul248(a, qweight, scales, qzeros, g_idx, bits, maxq)
                if transpose:
                    a = torch.randn(m, n, dtype=torch.float16, device='cuda')
                    transpose_matmul248(a, qweight, scales, qzeros, g_idx, bits, maxq)
    del kn_values
