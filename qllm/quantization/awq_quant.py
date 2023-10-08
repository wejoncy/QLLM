import torch
import torch.nn as nn
import tqdm
import functools
from collections import defaultdict

from texttable import Texttable

from ..utils.comm_utils import clear_memory
from ..utils.modelutils import get_op_name, get_op_by_name, set_op_by_name

from .quant_frame_base import QuantFrameBase
from ..utils import find_layers
from ._awq_quantizer import InternalAWQuantizer, pseudo_quantize_tensor, USE_ACCUMULATE_BATCH


def scale_activations(module):
    param = next(module.parameters())
    dtype = param.dtype
    device = param.device
    if 'mptblock' in str(module.__class__.__name__).lower():
        if isinstance(module.ffn.act, ScaledActivation):
            return
        c = module.ffn.up_proj.out_features
        act = ScaledActivation(
            module.ffn.act,
            torch.ones(c, dtype=dtype, device=device)
        )
        set_op_by_name(module, "ffn.act", act)
    elif 'falcon' in str(module.__class__).lower():
        if isinstance(module.mlp.act, ScaledActivation):
            return
        c = module.mlp.dense_h_to_4h.out_features
        act = ScaledActivation(
            module.mlp.act,
            torch.ones(c, dtype=dtype, device=device)
        )
        set_op_by_name(module, "mlp.act", act)



class AWQQuant(QuantFrameBase):
    def __init__(self, args) -> None:
        super().__init__(args)
        self.auto_scale = True
        self.auto_clip = True
        self.q_config = {
            "zero_point": True,  # by default True
            "q_group_size": args.groupsize,  # whether to use group quantization

        }

    def hijack_internal_block(self, named_linears, layer_block, inps, layer_kwargs):
        dev = next(layer_block.parameters()).device
        # firstly, get input features of all linear layers

        def cache_input_hook(m, x, y, name, feat_dict):
            x = x[0]
            x = x.detach().cpu()
            feat_dict[name].append(x)

        input_feat = defaultdict(list)
        handles = []
        for name in named_linears:
            handles.append(named_linears[name].register_forward_hook(
                functools.partial(cache_input_hook, name=name, feat_dict=input_feat)))
        # in case multi-gpu
        # get output as next layer's input
        if not USE_ACCUMULATE_BATCH:
            inps = inps.to(dev)
            outs = layer_block(inps, **layer_kwargs)[0]
        else:
            outs = []
            for input_tensor in inps:
                input_tensor = input_tensor.unsqueeze(0)
                input_tensor = input_tensor.to(dev)
                outs.append(layer_block(input_tensor, **layer_kwargs)[0])
            outs = torch.concat(outs, dim=0)
        for h in handles:
            h.remove()
        # now solve for scaling and clipping
        input_feat = {k: torch.cat(v, dim=0) for k, v in input_feat.items()}

        # Clear GPU memory
        clear_memory()
        return outs, input_feat

    def _apply_quant(self, model, named_linears, quantizers, state_dict_prefix, version="GEMM"):
        for name, linear_layer in named_linears.items():
            # NOTE: small regression in perplexity if linear layer uses .cpu().float()
            linear_layer = linear_layer.cuda().half()

            linear_layer.weight.data, scales, zeros = pseudo_quantize_tensor(
                linear_layer.weight.data,
                n_bit=self.args.wbits,
                q_config=self.q_config,
                get_scale_zp=True,
            )
            #get_op_name(model, linear_layer)
            layer_key = f"{state_dict_prefix}.{name}"
            quantizers[layer_key] = (
                None, scales.cpu(), zeros.cpu(), None, self.args.wbits, self.args.groupsize)

            clear_memory()

    @torch.no_grad()
    def quantize(self, model, dataloader, dev):
        args = self.args
        model = self.prepare(model)
        state_dict_prefix = self.extract_prefix(model)
        inps, outs, attention_layers, layer_kwargs = self.hijack_block_inputs(model, dataloader, args, dev)
        layer_kwargs['attention_mask'] = layer_kwargs['attention_mask'].expand(len(dataloader), -1, -1, -1)
        print('Ready.')

        quantizers = {}
        awq_results = {
            "scale": [],
            "clip": [],
        }
        # solve layer by layer
        for i in tqdm.tqdm(range(len(attention_layers)), desc="Running AWQ..."):
            layer = attention_layers[i]
            layer = layer.cuda()
            named_linears = find_layers(layer, self.quant_layers)
            inps, input_feat = self.hijack_internal_block(named_linears, layer, inps, layer_kwargs)

            in_quantizer = InternalAWQuantizer()
            in_quantizer.configure(args.wbits, self.q_config, self.auto_scale, self.auto_clip)

            in_quantizer.fast_quant_layer(layer_kwargs, input_feat, layer, attention_layers, i)

            layer = layer.cpu()
            # Haotian: check activation replacement
            clear_memory(input_feat)
            self._apply_quant(model, named_linears, quantizers, f"{state_dict_prefix}.{i}")
        # real_quantize_model_weight(attention_layers, args.wbits, self.q_config)
        return quantizers
