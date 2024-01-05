import torch
import torch.nn as nn
import numpy as np


class InternalHQQQuantizer(nn.Module):

    def __init__(self, layer: nn.Module):
        super(InternalHQQQuantizer, self).__init__()
        self.scale_quant_params = None
        self.zero_quant_params = None
        self.layer = layer


    def configure(self, bits: int, channel_wise: bool = True, group_size: int = 64,
                  optimize: bool = False, round_zero: bool = False, axis: int = 0):
        self.bits = bits
        assert bits in [2, 3, 4, 8], "bits=" + str(bits) + " not supported."

        self.channel_wise = channel_wise
        self.group_size = group_size
        self.optimize = optimize
        self.round_zero = round_zero
        self.axis = axis

    # Proximal solver || W - dequantize(quantize(W))||_p^p

    @torch.inference_mode()
    def optimize_weights_proximal(self, tensor, scale, zero, min_max, axis=0, device='cuda', opt_params={'lp_norm': 0.7, 'beta': 1e1, 'kappa': 1.01, 'iters': 20}, verbose=False):
        lp_norm, beta, kappa, iters = opt_params['lp_norm'], opt_params['beta'], opt_params['kappa'], opt_params['iters']

        dtype = torch.float16 if (device == 'cuda') else torch.float32
        W_f = tensor.to(dtype).to(device)
        scale = scale.to(dtype).to(device)
        zero = zero.to(dtype).to(device)

        if (lp_norm == 1):
            shrink_op = lambda x, beta: torch.sign(x) * torch.nn.functional.relu(torch.abs(x) - 1. / beta)
        else:
            shrink_op = lambda x, beta, p=lp_norm: torch.sign(x) * torch.nn.functional.relu(torch.abs(x) - (1. / beta) * torch.pow(torch.abs(x), p - 1))

        best_error = 1e4
        for i in range(iters):
            W_q = torch.round(W_f * scale + zero).clamp(min_max[0], min_max[1])
            W_r = (W_q - zero) / scale
            W_e = shrink_op(W_f - W_r, beta)
            zero = torch.mean(W_q - (W_f - W_e) * scale, axis=axis, keepdim=True)
            beta *= kappa

            current_error = float(torch.abs(W_f - W_r).mean())
            if (verbose):
                print(i, np.round(current_error, 6))
            if (current_error < best_error):
                best_error = current_error
            else:
                break

        scale = scale.to(tensor.device)
        zero = zero.to(tensor.device)
        del W_f, W_q, W_r, W_e
        torch.cuda.empty_cache()

        return scale, zero

    def quantize(self):
        tensor = self.layer.weight
        nbits = self.bits
        channel_wise = self.channel_wise,
        group_size = self.group_size
        optimize = self.optimize,
        round_zero = self.round_zero,
        axis = self.axis

        assert axis in [0, 1], "axis should be either 0 or 1"
        if (group_size is not None):
            assert tensor.shape[axis] % group_size == 0, "group_size should be divisble by the total tensor dimensions."

        W = tensor.float()
        shape = W.shape

        # Reshape for grouping
        if ((group_size is not None) and channel_wise):
            W = W.reshape([-1, group_size]) if (axis == 1) else W.reshape([group_size, -1])

        # Get min/max values
        if (channel_wise == False):
            _min, _max = W.min(), W.max()
            optimize = False
        else:
            _min = W.min(axis=axis, keepdim=True)[0]
            _max = W.max(axis=axis, keepdim=True)[0]

        max_v = 2**nbits - 1
        min_v = 0
        min_max = [min_v, max_v]

        # Note: here we work with the inverse of the scale to avoid division and quantize instead via W*scale + zero, the scale is inverted later on.
        scale = (max_v / (_max - _min)).clamp(max=2e4)  # clamp to avoid half-precision problems
        zero = -_min * scale

        if (round_zero): zero = torch.round(zero)

        # Fine-tune weights
        if (optimize): 
            scale, zero = self.optimize_weights_proximal(
                tensor=W, scale=scale, zero=zero, min_max=min_max, axis=axis)
		#Quantize
        W_q  = torch.round(W*scale + zero).clamp(min_max[0], min_max[1])
        self.layer.weight.data = ((W_q- zero)/scale).reshape(shape).type(tensor.dtype)
        scale = 1.0/scale
        # cleanup
        del W, _min, _max
        torch.cuda.empty_cache()

        if axis == 1:
            scale = scale.reshape(shape[0], -1)
            zero = zero.reshape(shape[0], -1)
        else:
            scale = scale.reshape(-1, shape[-1])
            zero = zero.reshape(-1, shape[-1])
        return scale.cpu(), zero.cpu()
    
    def free(self):
        del self.layer
        torch.cuda.empty_cache()
