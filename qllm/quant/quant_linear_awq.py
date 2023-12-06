import math
import torch
import torch.nn as nn
from .compress_weight import CompressWeight
import importlib


def has_awq_inference_engine():
    return (importlib.util.find_spec("awq_inference_engine") is not None and
            torch.cuda.get_device_properties(0).major >= 8)


def is_the_machine_support_awq_engine(nbits):
    return has_awq_inference_engine() and nbits == 4


if not has_awq_inference_engine():
    print("awq_inference_engine not found, please compile it first.")
else:
    import awq_inference_engine


def make_divisible(c, divisor):
    return (c + divisor - 1) // divisor


def calculate_zeros_width(in_features, group_size=128, pack_num=8):
    if group_size >= 128:
        size_multiplier = 1
    elif group_size == 64:
        size_multiplier = 2
    elif group_size == 32:
        size_multiplier = 4
    else:
        raise NotImplementedError

    base_width = make_divisible(in_features // group_size, pack_num)
    base_width = make_divisible(base_width, size_multiplier) * size_multiplier
    return base_width


class WQLinear_GEMM(nn.Module, CompressWeight):
    def __init__(self, w_bit, group_size, in_features, out_features, bias):
        super().__init__()

        if w_bit not in [4]:
            raise NotImplementedError("Only 4-bit are supported for now.")

        self.infeatures = in_features
        self.outfeatures = out_features
        self.w_bit = w_bit
        self.group_size = group_size if group_size != -1 else in_features
        self.groupsize = self.group_size
        self.bits = w_bit
        self.orig_fp_weight = None

        # quick sanity check (make sure aligment)
        assert self.infeatures % self.group_size == 0
        assert out_features % (32 // self.w_bit) == 0

        self.g_idx = torch.tensor([i // group_size for i in range(in_features)], dtype=torch.int32)
        dev = torch.device('cpu')
        self.register_buffer('qweight', torch.zeros((in_features, out_features //
                             (32 // self.w_bit)), dtype=torch.int32, device=dev))
        self.register_buffer('qzeros', torch.zeros((in_features // self.group_size,
                             out_features // (32 // self.w_bit)), dtype=torch.int32, device=dev))
        self.register_buffer('scales', torch.zeros(
            (in_features // self.group_size, out_features), dtype=torch.float16, device=dev))
        if bias:
            self.register_buffer('bias', torch.zeros((out_features), dtype=torch.float16, device=dev))
        else:
            self.bias = None

    def reorder_int_tensor(self, int_tensor):
        compress_ratio = (32 // self.bits)
        assert int_tensor.shape[-1] % compress_ratio == 0
        if self.w_bit == 4:
            order_map = [0, 2, 4, 6, 1, 3, 5, 7]
        else:
            raise NotImplementedError("Only 4-bit are supported for now.")
        order_tensor = torch.tensor(
            order_map, dtype=torch.int32, device=int_tensor.device).reshape(1, -1)
        order_tensor = order_tensor.repeat(
            int_tensor.shape[1]//compress_ratio, 1)
        order_tensor = order_tensor + torch.arange(0, int_tensor.shape[1],
                                                   compress_ratio, dtype=torch.int32, device=int_tensor.device).reshape(-1, 1)
        order_tensor = order_tensor.reshape(-1)
        int_tensor = int_tensor[:, order_tensor]
        int_tensor = int_tensor.T.contiguous()
        return int_tensor
        
    def reverse_reorder_int_tensor(self, int_tensor):
        int_tensor = int_tensor.T.contiguous()
        compress_ratio = (32 // self.bits)
        assert int_tensor.shape[-1] % compress_ratio == 0
        if self.w_bit == 4:
            order_map = [0, 2, 4, 6, 1, 3, 5, 7]
        else:
            raise NotImplementedError("Only 4-bit are supported for now.")
        order_tensor = torch.tensor(
            order_map, dtype=torch.int32, device=int_tensor.device).reshape(1, -1)
        order_tensor = order_tensor.repeat(
            int_tensor.shape[1]//compress_ratio, 1)
        order_tensor = order_tensor + torch.arange(0, int_tensor.shape[1],
                                                   compress_ratio, dtype=torch.int32, device=int_tensor.device).reshape(-1, 1)
        order_tensor = order_tensor.reshape(-1)

        reverse_order_tensor = torch.arange(order_tensor.shape[0]).cuda()[order_tensor]
        reverse_order_tensor = reverse_order_tensor[order_tensor]
        int_tensor = int_tensor[:, reverse_order_tensor]
        return int_tensor

    @torch.no_grad()
    def forward(self, x):
        out_shape = x.shape[:-1] + (self.outfeatures, )
        out = awq_inference_engine.gemm_forward_cuda(
            x.reshape(-1, x.shape[-1]), self.qweight, self.scales, self.qzeros, 8)
        out = out + self.bias if self.bias is not None else out
        return out.reshape(out_shape)

    def extra_repr(self) -> str:
        return 'infeatures={}, outfeatures={}, bias={}, w_bit={}, group_size={}'.format(
            self.infeatures, self.outfeatures, self.bias is not None, self.w_bit, self.group_size
        )


class WQLinear_GEMV(nn.Module):
    def __init__(self, w_bit, group_size, in_features, out_features, bias):
        super().__init__()

        if w_bit not in [4]:
            raise NotImplementedError("Only 4-bit are supported for now.")
        dev = torch.device('cpu')
        self.in_features = in_features
        self.out_features = out_features
        self.w_bit = w_bit
        self.group_size = group_size if group_size != -1 else in_features
        self.split_k_iters = 8

        # quick sanity check (make sure aligment)
        assert self.in_features % self.group_size == 0
        assert out_features % (32 // self.w_bit) == 0
        pack_num = (32 // self.w_bit)

        self.register_buffer('qweight', torch.zeros(
            (out_features, in_features // pack_num), dtype=torch.int32, device=dev))
        self.register_buffer('qzeros', torch.zeros((out_features, calculate_zeros_width(
            in_features, self.group_size)), dtype=torch.int32, device=dev))
        self.register_buffer('scales', torch.zeros((out_features, calculate_zeros_width(
            in_features, self.group_size) * pack_num), dtype=torch.float16, device=dev))
        if bias:
            self.register_buffer('bias', torch.zeros((out_features), dtype=torch.float16, device=dev))
        else:
            self.bias = None

    def pack_gpu(cls, linear, w_bit, group_size, init_only=False, scales=None, zeros=None):
        awq_linear = cls(w_bit, group_size, linear.in_features, linear.out_features,
                         linear.bias is not None, linear.weight.device)
        if init_only:  # just prepare for loading sd
            return awq_linear

        # need scales and zeros info for real quantization
        assert scales is not None and zeros is not None
        scale_zeros = zeros * scales

        pack_num = 32 // awq_linear.w_bit
        qscales = torch.zeros(
            (scales.shape[0], calculate_zeros_width(linear.in_features, group_size) * pack_num),
            dtype=torch.float16,
            device=scales.device
        )
        qscales[:, :scales.shape[1]] = scales
        awq_linear.scales = qscales
        if linear.bias is not None:
            awq_linear.bias = linear.bias.clone().half()

        intweight = []
        for idx in range(awq_linear.in_features):
            intweight.append(torch.round(
                (linear.weight.data[:, idx] + scale_zeros[:, idx // group_size]) / awq_linear.scales[:, idx // group_size]).to(torch.int)[:, None])
        intweight = torch.cat(intweight, dim=1)
        intweight = intweight.to(dtype=torch.int32)
        qweight = torch.zeros((intweight.shape[0], intweight.shape[1] // 32 *
                              awq_linear.w_bit), dtype=torch.int32, device=intweight.device)

        for col in range(intweight.shape[1] // pack_num):
            if awq_linear.w_bit == 4:
                order_map = [0, 1, 2, 3, 4, 5, 6, 7]
            else:
                raise NotImplementedError("Only 4-bit are supported for now.")
            for i in range(pack_num):
                qweight_col = intweight[:, col * pack_num + order_map[i]]
                qweight[:, col] |= qweight_col << (i * awq_linear.w_bit)
        awq_linear.qweight = qweight

        zeros = zeros.to(dtype=torch.int32)
        qzeros = torch.zeros(
            (zeros.shape[0], calculate_zeros_width(linear.in_features, group_size)),
            dtype=torch.int32,
            device=zeros.device,
        )

        for col in range((zeros.shape[1] + pack_num - 1) // pack_num):
            if awq_linear.w_bit == 4:
                order_map = [0, 1, 2, 3, 4, 5, 6, 7]
            else:
                raise NotImplementedError("Only 4-bit are supported for now.")
            for i in range(pack_num):
                if col * pack_num + order_map[i] >= zeros.shape[1]:
                    continue
                qzero_col = zeros[:, col * pack_num + order_map[i]]
                qzeros[:, col] |= qzero_col << (i * awq_linear.w_bit)
        awq_linear.qzeros = qzeros
        return awq_linear

    @torch.no_grad()
    def forward(self, x):
        out_shape = x.shape[:-1] + (self.out_features, )
        inputs = x.reshape(-1, x.shape[-1])

        if inputs.shape[0] > 8:
            out = awq_inference_engine.gemmv2_forward_cuda(
                inputs, self.qweight, self.scales, self.qzeros, self.group_size, self.split_k_iters)
        else:
            out = awq_inference_engine.gemv_forward_cuda(
                inputs, self.qweight, self.scales, self.qzeros, self.group_size)

        out = out + self.bias if self.bias is not None else out
        return out.reshape(out_shape)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}, w_bit={}, group_size={}'.format(
            self.in_features, self.out_features, self.bias is not None, self.w_bit, self.group_size
        )
