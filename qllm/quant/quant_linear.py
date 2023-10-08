import math
import os
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import custom_bwd, custom_fwd
import importlib
from .compress_weight import CompressWeight, general_unpack_on_row


try:
    has_module_dequantize_nbit_ops = importlib.util.find_spec("dequantize_nbit_ops")

    if has_module_dequantize_nbit_ops == False:
        import subprocess
        import sys
        subprocess.check_call([sys.executable, '-m', 'pip', 'install',
                              'git+https://github.com/wejoncy/DQOps.git@0.1#egg=dequantize_nbit_ops'])
        import dequantize_nbit_ops  # cuda dequant
        has_module_dequantize_nbit_ops = True
except:
    print("torch implementation of dequantization would be used")
    pass

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


class QuantLinearTorchFunction(torch.autograd.Function):
    @staticmethod
    def symbolic(g,  x, qself_qweight, qself_scales, qself_qzeros, qself_sbias, qself_g_idx, bits, groupsize, in_features):
        return g.op("com.microsoft::QuantNbitsGemm", x, qself_qweight, qself_scales, qself_qzeros, qself_sbias, qself_g_idx,
                    outputs=1, in_features_i=in_features, bits_i=bits, groupsize_i=groupsize)

    @staticmethod
    def forward(ctx, input, qweight, scales, qzeros, sbias, g_idx, bits, groupsize, in_features):
        if os.getenv("export_onnx", "0") == "1":
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
    def symbolic(g, qself_qweight, scales, qzeros, groupsize, bits, in_features):
        return g.op("com.microsoft::DequantizeAndUnpackWeight", qself_qweight, scales, qzeros,
                    outputs=1, groupsize_i=groupsize, bits_i=bits, in_features_i=in_features)

    @staticmethod
    def forward(ctx, qweight, scales, qzeros, groupsize, bits, in_features):
        if has_module_dequantize_nbit_ops and qweight.is_cuda:
            return dequantize_nbit_ops.dequant(qweight, scales, qzeros, groupsize, bits, in_features)
        scales = scales.reshape(-1, 1, scales.shape[-1])
        if bits in [2, 4, 8]:
            wf = torch.tensor(list(range(0, 32, bits)), dtype=torch.int32, device=qweight.device).unsqueeze(0)
            # expand is removed as torch will auto broadcast to relavant dimension
            zeros = torch.bitwise_right_shift(torch.unsqueeze(qzeros, 2), wf.unsqueeze(0)
                                              ).to(torch.int16 if bits == 8 else torch.int8)
            zeros = torch.bitwise_and(zeros, (2 ** bits) - 1)
            # zeros = zeros + 1
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

        scale_zeros = zeros * scales
        weight = weight.reshape(-1, groupsize, weight.shape[-1])
        weight = (scales * weight - scale_zeros.half())

        # weight = (scales * (weight - zeros))
        weight = weight.reshape(-1, weight.shape[2])
        return weight


def QuantLinearTorchFunction_forward(input, qweight, scales, qzeros, g_idx, bits, groupsize, in_features):
    weight = DequantAndUnpack().apply(qweight, scales, qzeros, groupsize, bits, in_features)
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
        self.oweight = None
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


    def forward(self, x):
        # would contiguous() here affect accuracy? the default should be contiguous().T
        # out =  torch.matmul(x.reshape(-1, x.shape[-1]),self.oweight.T.contiguous())
        out_shape = x.shape[:-1] + (self.outfeatures, )
        # out = QuantLinearFunction.apply(x.reshape(-1, x.shape[-1]), self.qweight, self.scales, self.qzeros, self.g_idx, self.bits, self.maxq)
        # sbias = self.bias if self.bias is not None else torch.tensor([0],dtype=torch.float16)
        # out = QuantLinearTorchFunction.apply(x, self.qweight, self.scales, self.qzeros, sbias, self.g_idx, self.bits, self.groupsize, self.infeatures)
        out = QuantLinearTorchFunction_forward(x, self.qweight, self.scales,
                                               self.qzeros, self.g_idx, self.bits, self.groupsize, self.infeatures)
        out = out + self.bias if self.bias is not None else out
        return out


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
