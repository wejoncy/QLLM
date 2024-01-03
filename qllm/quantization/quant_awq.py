import torch
import tqdm
import functools
from collections import defaultdict

from ..utils.comm_utils import clear_memory
from ..utils.modelutils import set_op_by_name

from .quant_frame_base import QuantFrameBase
from ..utils import find_layers
from ._awq_quantizer import (InternalAWQuantizer, pseudo_quantize_tensor,
                             USE_ACCUMULATE_BATCH, ScaledActivation)


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
        self.quant_config = {"zero_point": True, "q_group_size": args.groupsize,
                "w_bit": args.wbits, "method": 'awq'}

    def hijack_internal_block(self, named_linears, layer_block, inps, layer_kwargs):
        dev = next(layer_block.parameters()).device
        # firstly, get input features of all linear layers
        if "mixtral" in (layer_block).__class__.__name__.lower():
            named_linears.pop("block_sparse_moe.gate", None)
            named_linears = {**named_linears, "block_sparse_moe": layer_block.block_sparse_moe}
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
        if USE_ACCUMULATE_BATCH == -1:
            inps = inps.to(dev)
            outs = layer_block(inps, **layer_kwargs)[0]
        else:
            outs = []
            for start in range(0, len(inps), USE_ACCUMULATE_BATCH):
                end = min(start + USE_ACCUMULATE_BATCH, len(inps))
                single_x = inps[start:end].to(dev)
                outs.append(layer_block(single_x, **layer_kwargs)[0])
            for key in input_feat:
                input_feat[key] = [torch.cat(input_feat[key], dim=0)]
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
                q_config=self.quant_config,
                get_scale_zp=True,
            )
            # get_op_name(model, linear_layer)
            layer_key = f"{state_dict_prefix}.{name}"
            quantizers[layer_key] = (
                None, scales.cpu(), zeros.cpu(), None, self.args.wbits, self.args.groupsize)
            linear_layer.cpu()
            clear_memory(scales, zeros)

    @torch.no_grad()
    def do_quantize(self, model, dataloader, model_prefix, dev):
        args = self.args
        inps, outs, attention_layers, layer_kwargs = self.hijack_block_inputs(model, dataloader, model_prefix, dev)
        if USE_ACCUMULATE_BATCH == -1:
            run_batch = len(dataloader)
        else:
            run_batch = USE_ACCUMULATE_BATCH
        if layer_kwargs.get('attention_mask', None) is not None:
            layer_kwargs['attention_mask'] = layer_kwargs['attention_mask'].expand(run_batch, -1, -1, -1)        
        print('Ready.')

        quantizers = {}
        # solve layer by layer
        for i in tqdm.tqdm(range(len(attention_layers)), desc="Running AWQ..."):
            layer = attention_layers[i]
            layer = layer.cuda()
            named_linears = find_layers(layer, self.quant_layers)
            inps, input_feat = self.hijack_internal_block(named_linears, layer, inps, layer_kwargs)

            in_quantizer = InternalAWQuantizer()
            in_quantizer.configure(args.wbits, self.quant_config, self.auto_scale, self.auto_clip)

            in_quantizer.fast_quant_layer(layer_kwargs, input_feat, layer, attention_layers, i, model.__class__.__name__)
            self._apply_quant(model, named_linears, quantizers, f"{model_prefix}.{i}")

            layer = layer.cpu()
            # Haotian: check activation replacement
            clear_memory(input_feat)
        # real_quantize_model_weight(attention_layers, args.wbits, self.quant_config)
        
        return quantizers
