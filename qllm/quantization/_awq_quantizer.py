import numpy as np
import torch
import torch.nn as nn
import math
from ..utils.comm_utils import clear_memory
from ..utils.modelutils import get_op_name, get_op_by_name, set_op_by_name, ScaledLinear


USE_ACCUMULATE_BATCH = False



def get_model_specific_quant_layer(module, input_feat, module_kwargs):
    from transformers.models.opt.modeling_opt import OPTDecoderLayer
    from transformers.models.llama.modeling_llama import LlamaDecoderLayer

    scales_list = []  # return the searched scales

    if isinstance(module, OPTDecoderLayer):
        # attention input
        scales_list.append(dict(
            prev_op=module.self_attn_layer_norm,
            layers=[module.self_attn.q_proj,
                    module.self_attn.k_proj, module.self_attn.v_proj],
            inp=input_feat['self_attn.q_proj'],
            module2inspect=module.self_attn, kwargs=module_kwargs,
        ))
        # attn out
        scales_list.append(dict(
            prev_op=module.self_attn.v_proj,
            layers=[module.self_attn.out_proj],
            inp=input_feat['self_attn.out_proj'],
        ))
        # fc1
        scales_list.append(dict(
            prev_op=module.final_layer_norm,
            layers=[module.fc1],
            inp=input_feat['fc1'],
        ))
        # fc2
        scales_list.append(dict(
            prev_op=module.fc1,
            layers=[module.fc2],
            inp=input_feat['fc2'],
        ))

    elif isinstance(module, LlamaDecoderLayer):
        # attention input
        scales_list.append(dict(
            prev_op=module.input_layernorm,
            layers=[module.self_attn.q_proj,
                    module.self_attn.k_proj, module.self_attn.v_proj],
            inp=input_feat['self_attn.q_proj'],
            module2inspect=module.self_attn, kwargs=module_kwargs,
        ))
        # attn out
        # Please refer to https://github.com/mit-han-lab/llm-awq/pull/67#issue-1850622696
        if module.self_attn.v_proj.weight.shape == module.self_attn.o_proj.weight.shape:
            scales_list.append(dict(
                prev_op=module.self_attn.v_proj,
                layers=[module.self_attn.o_proj],
                inp=input_feat['self_attn.o_proj'],
            ))
        # fc1
        scales_list.append(dict(
            prev_op=module.post_attention_layernorm,
            layers=[module.mlp.gate_proj, module.mlp.up_proj],
            inp=input_feat['mlp.gate_proj'],
            module2inspect=module.mlp,
        ))
        # fc2
        scales_list.append(dict(
            prev_op=module.mlp.up_proj,
            layers=[module.mlp.down_proj],
            inp=input_feat['mlp.down_proj'],
        ))
    elif "mpt" in str(module.__class__).lower():
        # attention input
        scales_list.append(dict(
            prev_op=module.norm_1,
            layers=[module.attn.Wqkv],
            inp=input_feat['attn.Wqkv'],
            module2inspect=module.attn,
            kwargs=module_kwargs,
        ))

        # attn out
        scales_list.append(dict(
            prev_op=module.attn.Wqkv,
            layers=[module.attn.out_proj],
            inp=input_feat['attn.out_proj'],
        ))
        # fc1
        scales_list.append(dict(
            prev_op=module.norm_2,
            layers=[module.ffn.up_proj],
            inp=input_feat['ffn.up_proj'],
            module2inspect=module.ffn,
        ))
        # fc2
        scales_list.append(dict(
            prev_op=module.ffn.act,
            layers=[module.ffn.down_proj],
            inp=input_feat['ffn.down_proj'],
        ))
    else:
        raise NotImplementedError(f"{type(module)} not supported yet!")

    return scales_list

@torch.no_grad()
def get_weight_scale(weight, q_group_size=-1):
    org_shape = weight.shape
    if q_group_size > 0:
        weight = weight.view(-1, q_group_size)
    scale = weight.abs() / weight.abs().amax(dim=1, keepdim=True)
    scale = scale.view(org_shape)
    scale = scale.mean(0)
    return scale

@torch.no_grad()
def get_act_scale(x):
    return x.abs().view(-1, x.shape[-1]).mean(0)


@torch.no_grad()
def scale_ln_fcs(ln, fcs, scales):
    if not isinstance(fcs, list):
        fcs = [fcs]

    scales = scales.to(ln.weight.device)

    # debugging start even scales = 1 does not work?
    """
    scales = scales * 0
    scales = scales + 1
    """
    # debugging end

    ln.weight.div_(scales)
    if hasattr(ln, 'bias') and ln.bias is not None:
        ln.bias.div_(scales)

    for fc in fcs:
        fc.weight.mul_(scales.view(1, -1))

    for p in ln.parameters():
        assert torch.isnan(p).sum() == 0
    for fc in fcs:
        for p in fc.parameters():
            assert torch.isnan(p).sum() == 0


@torch.no_grad()
def scale_fc_fc(fc1, fc2, scales):
    assert isinstance(fc1, nn.Linear)
    assert isinstance(fc2, nn.Linear)
    # assert fc1.out_features == fc2.in_features

    scales = scales.to(fc1.weight.device)

    # fc1.weight.div_(scales.view(-1, 1))
    fc1.weight[-scales.size(0):].div_(scales.view(-1, 1))
    if fc1.bias is not None:
        fc1.bias.div_(scales.view(-1))

    fc2.weight.mul_(scales.view(1, -1))

    for p in fc1.parameters():
        assert torch.isnan(p).sum() == 0
    for p in fc2.parameters():
        assert torch.isnan(p).sum() == 0


@torch.no_grad()
def scale_gelu_fc(gelu, fc, scales):
    assert isinstance(gelu, nn.GELU) or isinstance(gelu, BloomGelu)
    assert isinstance(fc, nn.Linear)

    fc.weight.mul_(scales.view(1, -1).to(fc.weight.device))

    for p in fc.parameters():
        assert torch.isnan(p).sum() == 0


class ScaledActivation(nn.Module):
    def __init__(self, module, scales):
        super().__init__()
        self.act = module
        self.scales = nn.Parameter(scales.data)

    def forward(self, x):
        return self.act(x) / self.scales.view(1, 1, -1).to(x.device)


def apply_scale(module, scales_list, input_feat_dict=None):
    for prev_op_name, layer_names, scales in scales_list:
        # replace linear with scaled linear, so we don't have to modify the prev op
        #for name in layer_names:
        #    set_op_by_name(module, name, ScaledLinear(get_op_by_name(module, name), scales))
        #continue

        prev_op = get_op_by_name(module, prev_op_name)
        layers = [get_op_by_name(module, name) for name in layer_names]

        prev_op.cuda()
        for layer in layers:
            layer.cuda()
        scales.cuda()
        from transformers.models.llama.modeling_llama import LlamaRMSNorm

        if isinstance(prev_op, nn.Linear):
            assert len(layers) == 1
            scale_fc_fc(prev_op, layers[0], scales)
        elif isinstance(prev_op, (nn.LayerNorm, LlamaRMSNorm)):
            scale_ln_fcs(prev_op, layers, scales)
        elif isinstance(prev_op, nn.GELU):
            new_module = ScaledActivation(prev_op, scales)
            set_op_by_name(module, prev_op_name, new_module)
            scale_gelu_fc(prev_op, layers[0], scales)
        else:
            raise NotImplementedError(
                f"prev_op {type(prev_op)} not supported yet!")

        # apply the scaling to input feat if given; prepare it for clipping
        if input_feat_dict is not None:
            for layer_name in layer_names:
                inp = input_feat_dict[layer_name]
                inp.div_(scales.view(1, -1).to(inp.device))

        prev_op.cpu()
        for layer in layers:
            layer.cpu()
        scales.cpu()

# core quantization method (simulated quantization)
def pseudo_quantize_tensor(w, n_bit, q_config, inplace=False, get_scale_zp=False):
    zero_point, q_group_size= q_config.get("zero_point", True), q_config.get("q_group_size", -1)
    org_w_shape = w.shape
    if q_group_size > 0:
        assert org_w_shape[-1] % q_group_size == 0
        w = w.reshape(-1, q_group_size)
    assert w.dim() == 2
    if zero_point:
        max_val = w.amax(dim=1, keepdim=True)
        min_val = w.amin(dim=1, keepdim=True)
        max_int = 2 ** n_bit - 1
        min_int = 0
        scales = (max_val - min_val).clamp(min=1e-5) / max_int
        zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)
    else:  # we actually never used this
        assert min_val is None
        max_val = w.abs().amax(dim=1, keepdim=True)
        max_val = max_val.clamp(min=1e-5)
        max_int = 2 ** (n_bit - 1) - 1
        min_int = - 2 ** (n_bit - 1)
        scales = max_val / max_int
        zeros = 0

    assert torch.isnan(scales).sum() == 0
    assert torch.isnan(w).sum() == 0

    if inplace:
        ((w.div_(scales).round_().add_(zeros)).clamp_(
            min_int, max_int).sub_(zeros)).mul_(scales)
    else:
        w = (torch.clamp(torch.round(w / scales) +
                        zeros, min_int, max_int) - zeros) * scales
    assert torch.isnan(w).sum() == 0

    w = w.reshape(org_w_shape)

    if get_scale_zp:
        return w, scales.view(w.shape[0], -1), zeros.view(w.shape[0], -1)
    else:
        return w

# weight quantization
@torch.no_grad()
def auto_clip_layer(w, input_feat, n_bit, q_config,
                    n_grid=20,
                    max_shrink=0.5,
                    n_sample_token=512):
    assert w.dim() == 2
    org_w_shape = w.shape
    # w           [co, ci]      -> [co, 1, n_group, group size]
    # input_feat  [n_token, ci] -> [1, n_token, n_group, group size]
    group_size = q_config["q_group_size"] if q_config["q_group_size"] > 0 else w.shape[1]
    input_feat = input_feat.view(-1, input_feat.shape[-1])
    input_feat = input_feat.reshape(1, input_feat.shape[0], -1, group_size)
    input_feat = input_feat[:, 0::input_feat.shape[1] // n_sample_token]
    w = w.reshape(w.shape[0], 1, -1, group_size)

    oc_batch_size = 256 if w.shape[0] % 256 == 0 else 64  # prevent OOM
    assert w.shape[0] % oc_batch_size == 0
    w_all = w
    best_max_val_all = []

    for i_b in range(w.shape[0] // oc_batch_size):
        w = w_all[i_b * oc_batch_size: (i_b + 1) * oc_batch_size]

        org_max_val = w.abs().amax(dim=-1, keepdim=True)  # co, 1, n_group, 1

        best_max_val = org_max_val.clone()
        min_errs = torch.ones_like(org_max_val) * 1e9
        input_feat = input_feat.to(w.device)
        org_out = (input_feat * w).sum(dim=-1)  # co, n_token, n_group

        for i_s in range(int(max_shrink * n_grid)):
            max_val = org_max_val * (1 - i_s / n_grid)
            min_val = - max_val
            cur_w = torch.clamp(w, min_val, max_val)
            q_w = pseudo_quantize_tensor(cur_w, n_bit, q_config)
            cur_out = (input_feat * q_w).sum(dim=-1)

            # co, 1, n_group, 1
            err = (cur_out - org_out).pow(2).mean(dim=1).view(min_errs.shape)
            del cur_w
            del cur_out
            cur_best_idx = err < min_errs
            min_errs[cur_best_idx] = err[cur_best_idx]
            best_max_val[cur_best_idx] = max_val[cur_best_idx]
        best_max_val_all.append(best_max_val)

    best_max_val = torch.cat(best_max_val_all, dim=0)

    del input_feat
    clear_memory(org_out)

    return best_max_val.squeeze(1)


@torch.no_grad()
def auto_clip_block(module,
                    w_bit, q_config,
                    input_feat):

    named_linears = {name: m for name,
                     m in module.named_modules() if isinstance(m, nn.Linear)}

    clip_list = []
    for name in named_linears:
        # due to qk bmm, it is hard to clip precisely
        if any([_ in name for _ in ["q_", "k_", "query", "key", "Wqkv"]]):
            continue
        named_linears[name].cuda()
        max_val = auto_clip_layer(
            named_linears[name].weight, input_feat[name], n_bit=w_bit, q_config=q_config)
        clip_list.append((name, max_val))
        named_linears[name].cpu()
    return clip_list


@torch.no_grad()
def apply_clip(module, clip_list):
    for name, max_val in clip_list:
        layer = get_op_by_name(module, name)
        layer.cuda()
        max_val = max_val.to(layer.weight.device)
        org_shape = layer.weight.shape
        layer.weight.data = layer.weight.data.reshape(*max_val.shape[:2], -1)
        layer.weight.data = torch.clamp(layer.weight.data, -max_val, max_val)
        layer.weight.data = layer.weight.data.reshape(org_shape)
        layer.cpu()

class InternalAWQuantizer(nn.Module):
    def __init__(self):
        super(InternalAWQuantizer, self).__init__()
        self.w_bit = None
        self.q_config = None
        self.auto_scale = None
        self.auto_clip = None
    
    def configure(self, w_bit, q_config, auto_scale=True, auto_clip=True):
        self.w_bit = w_bit
        self.q_config = q_config
        self.auto_scale = auto_scale
        self.auto_clip = auto_clip


    @torch.no_grad()
    def auto_scale_block(self, module, module_kwargs, input_feat):
        def w_quantize_func(p): 
            return pseudo_quantize_tensor(p, self.w_bit, self.q_config).detach()

        if "use_cache" in module_kwargs:
            module_kwargs.pop("use_cache")

        # find the best scale ratio
        def _search_module_scale(block, linears2scale: list, x, kwargs={}):
            # w: co, ci
            # x: n, ci
            weight = torch.cat([_m.weight for _m in linears2scale], dim=0)
            w_max = get_weight_scale(
                weight, q_group_size=self.q_config.get("q_group_size", -1))
            # Clear GPU memory
            clear_memory(weight)

            x = x.to(next(block.parameters()).device)
            with torch.no_grad():
                if USE_ACCUMULATE_BATCH:
                    org_outs = []
                    for single_x in x:
                        single_x = single_x.unsqueeze(0)
                        org_outs.append(block(single_x, **kwargs)[0])
                    org_out = torch.cat(org_outs, dim=0)
                else:
                    org_out = block(x, **kwargs)
                    if isinstance(org_out, tuple):
                        org_out = org_out[0]

            x_max = get_act_scale(x)

            best_error = float('inf')
            best_ratio = -1
            best_scales = None

            n_grid = 20
            history = []

            org_sd = {k: v.cpu() for k, v in block.state_dict().items()}
            for ratio in range(n_grid):
                ratio = ratio * 1 / n_grid
                scales = (x_max.pow(ratio) / w_max.pow(1-ratio)
                        ).clamp(min=1e-4).view(-1)
                scales = scales / (scales.max() * scales.min()).sqrt()
                for fc in linears2scale:
                    fc.weight.mul_(scales.view(1, -1).to(fc.weight.device))
                    fc.weight.data = w_quantize_func(
                        fc.weight.data) / (scales.view(1, -1))
                if USE_ACCUMULATE_BATCH:
                    outs = []
                    for single_x in x:
                        single_x = single_x.unsqueeze(0)
                        outs.append(block(single_x, **kwargs)[0])
                    out = torch.concat(outs, dim=0)
                else:
                    out = block(x, **kwargs)
                    if isinstance(out, tuple):
                        out = out[0]

                # float prevents overflow
                loss = (org_out - out).float().pow(2).mean().item()
                history.append(loss)
                is_best = loss < best_error
                if is_best:
                    best_error = loss
                    best_ratio = ratio
                    best_scales = scales
                block.load_state_dict(org_sd)
            if best_ratio == -1:
                print(history)
                raise Exception
            # print(best_ratio)
            best_scales = best_scales.view(-1)

            assert torch.isnan(best_scales).sum() == 0, best_scales
            return best_scales.detach()

        def _auto_get_scale(prev_op, layers, inp, module2inspect=None, kwargs={}):
            # module2inspect: if given, we will check the output diff of this module instead of layers
            if module2inspect is None:
                assert len(layers) == 1
                module2inspect = layers[0]

            scales = _search_module_scale(module2inspect, layers, inp, kwargs)
            scales = scales.detach().cpu()
            # prev_op_name, [layer_name], scale
            return (get_op_name(module, prev_op), tuple([get_op_name(module, m) for m in layers]), scales)

        sub_modules = get_model_specific_quant_layer(
            module=module, input_feat=input_feat, module_kwargs=module_kwargs)
        # return the searched scales
        scales_list = [_auto_get_scale(**sub_module) for sub_module in sub_modules]

        return scales_list
    
    def fast_quant_layer(self, layer_kwargs, input_feat, layer, attention_layers, i):
        if self.auto_scale:  # if it applies, we should also modify the input_feat with scales
            scales_list = self.auto_scale_block(layer, layer_kwargs, input_feat=input_feat,)
            # apply_scale(layer, scales_list, input_feat_dict=input_feat)
            apply_scale(attention_layers[i], scales_list, input_feat_dict=input_feat)
            # append prefix to make names global
            #awq_results["scale"] += append_str_prefix(scales_list, get_op_name(model, layer) + ".")

            # Clear GPU memory
            clear_memory()

        if self.auto_clip:
            clip_list = auto_clip_block(layer, w_bit=self.w_bit, q_config=self.q_config, input_feat=input_feat)
            apply_clip(layer, clip_list)
            # append prefix to make names global
            #awq_results["clip"] += append_str_prefix(clip_list, get_op_name(model, layer) + ".")

            # Clear GPU memory
            clear_memory()
