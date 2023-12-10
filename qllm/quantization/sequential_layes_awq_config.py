import torch.nn as nn


def get_aquila_layers(module:nn.Module, input_feat, module_kwargs):
    layers = []
    # attention input
    layers.append(dict(
        prev_op=module.input_layernorm,
        layers=[module.self_attn.q_proj,
                module.self_attn.k_proj, module.self_attn.v_proj],
        inp=input_feat['self_attn.q_proj'],
        module2inspect=module.self_attn, kwargs=module_kwargs,
    ))

    # attention out
    # Please refer to https://github.com/mit-han-lab/llm-/pull/67#issue-1850622696
    if module.self_attn.v_proj.weight.shape == module.self_attn.o_proj.weight.shape:
        layers.append(dict(
            prev_op=module.self_attn.v_proj,
            layers=[module.self_attn.o_proj],
            inp=input_feat['self_attn.o_proj'],
        ))
    
    # linear 1
    layers.append(dict(
        prev_op=module.post_attention_layernorm,
        layers=[module.mlp.gate_proj, module.mlp.up_proj],
        inp=input_feat['mlp.gate_proj'],
        module2inspect=module.mlp,
    ))

    # linear 2
    layers.append(dict(
        prev_op=module.mlp.up_proj,
        layers=[module.mlp.down_proj],
        inp=input_feat['mlp.down_proj'],
    ))

    return layers

def get_bloom_layer(module: nn.Module, input_feat, module_kwargs):
        layers = []

        # attention input
        layers.append(dict(
            prev_op=module.input_layernorm,
            layers=[module.self_attention.query_key_value],
            inp=input_feat['self_attention.query_key_value'],
            module2inspect=module, kwargs=module_kwargs,
        ))
        # attention out
        # Please refer to https://github.com/mit-han-lab/llm-/issues/2#issuecomment-1606297469
        """
        scales_list.append(_auto_get_scale(
            prev_op=module.self_attention.query_key_value,
            layers=[module.self_attention.dense],
            inp=input_feat['self_attention.dense'],
        ))
        """
        # linear 1
        layers.append(dict(
            prev_op=module.post_attention_layernorm,
            layers=[module.mlp.dense_h_to_4h],
            inp=input_feat['mlp.dense_h_to_4h'],
            module2inspect=module, kwargs=module_kwargs,
        ))
        # linear 2
        layers.append(dict(
            prev_op=module.mlp.gelu_impl,
            layers=[module.mlp.dense_4h_to_h],
            inp=input_feat['mlp.dense_4h_to_h'],
        ))

        return layers

def get_falcon_layers(module: nn.Module, input_feat, module_kwargs):
        layers = []
        
        # Falcon 7B (older architecture)
        if module.config.num_attention_heads == 71:
            # linear 1 + attention
            layers.append(dict(
                prev_op=module.input_layernorm,
                layers=[module.mlp.dense_h_to_4h, module.self_attention.query_key_value],
                inp=input_feat['self_attention.query_key_value'],
                module2inspect=module,
                kwargs=module_kwargs,
            ))

        # Falcon 40B (newer architecture)
        else:
            # linear 1 + attention
            layers.append(dict(
                prev_op=module.ln_attn,
                layers=[module.self_attention.query_key_value],
                inp=input_feat['self_attention.query_key_value'],
                module2inspect=module,
                kwargs=module_kwargs,
            ))

            # linear 2
            layers.append(dict(
                prev_op=module.ln_mlp,
                layers=[module.mlp.dense_h_to_4h],
                inp=input_feat['mlp.dense_h_to_4h'],
                module2inspect=module,
                kwargs=module_kwargs,
            ))

        return layers
def get_bigcode_layers(module:nn.Module, input_feat, module_kwargs):
        layers = []

        # attention input
        layers.append(dict(
            prev_op=module.ln_1,
            layers=[module.attn.c_attn],
            inp=input_feat['attn.c_attn'],
            module2inspect=module.attn,
            kwargs=module_kwargs
        ))
        
        # linear 1
        layers.append(dict(
            prev_op=module.ln_2,
            layers=[module.mlp.c_fc],
            inp=input_feat['mlp.c_fc'],
            module2inspect=module.mlp
        ))

        # linear 2
        layers.append(dict(
            prev_op=module.mlp.act,
            layers=[module.mlp.c_proj],
            inp=input_feat['mlp.c_proj']
        ))

        return layers

def get_neox_layers(module: nn.Module, input_feat, module_kwargs):
        layers = []

        # attention input
        layers.append(dict(
            prev_op=module.input_layernorm,
            layers=[module.attention.query_key_value],
            inp=input_feat['attention.query_key_value'],
        ))

        # attention out
        # Please refer to https://github.com/mit-han-lab/llm-/issues/2#issuecomment-1606297469
        """
        layers.append(dict(
            prev_op=module.attention.query_key_value,
            layers=[module.attention.dense],
            inp=input_feat['attention.dense'],
        ))
        """

        # linear 1
        layers.append(dict(
            prev_op=module.post_attention_layernorm,
            layers=[module.mlp.dense_h_to_4h],
            inp=input_feat['mlp.dense_h_to_4h'],
        ))

        # linear 2
        layers.append(dict(
            prev_op=module.mlp.act,
            layers=[module.mlp.dense_4h_to_h],
            inp=input_feat['mlp.dense_4h_to_h'],
        ))

        return layers

def get_gptj_layers(module: nn.Module, input_feat, module_kwargs):
        layers = []

        # attention input + linear 1
        layers.append(dict(
            prev_op=module.ln_1,
            layers=[module.attn.q_proj,
                    module.attn.k_proj, module.attn.v_proj, module.mlp.fc_in],
            inp=input_feat['attn.q_proj'],
            module2inspect=module,
            kwargs=module_kwargs
        ))

        # attention out
        layers.append(dict(
            prev_op=module.attn.v_proj,
            layers=[module.attn.out_proj],
            inp=input_feat['attn.out_proj'],
        ))

        # linear 2
        layers.append(dict(
            prev_op=module.mlp.act,
            layers=[module.mlp.fc_out],
            inp=input_feat['mlp.fc_out'],
        ))

        return layers

def get_mistray_layers(module: nn.Module, input_feat, module_kwargs):
        layers = []

        # attention input
        layers.append(dict(
            prev_op=module.input_layernorm,
            layers=[module.self_attn.q_proj,
                    module.self_attn.k_proj, module.self_attn.v_proj],
            inp=input_feat['self_attn.q_proj'],
            module2inspect=module.self_attn, kwargs=module_kwargs,
        ))

        # attention out
        # Please refer to https://github.com/mit-han-lab/llm-/pull/67#issue-1850622696
        if module.self_attn.v_proj.weight.shape == module.self_attn.o_proj.weight.shape:
            layers.append(dict(
                prev_op=module.self_attn.v_proj,
                layers=[module.self_attn.o_proj],
                inp=input_feat['self_attn.o_proj'],
            ))
        
        # linear 1
        layers.append(dict(
            prev_op=module.post_attention_layernorm,
            layers=[module.mlp.gate_proj, module.mlp.up_proj],
            inp=input_feat['mlp.gate_proj'],
            module2inspect=module.mlp,
        ))

        # linear 2
        layers.append(dict(
            prev_op=module.mlp.up_proj,
            layers=[module.mlp.down_proj],
            inp=input_feat['mlp.down_proj'],
        ))

        return layers
def get_qwen_layers(module: nn.Module, input_feat, module_kwargs):
        layers = []

        # attention
        layers.append(
            dict(
                prev_op=module.ln_1,
                layers=[module.attn.c_attn],
                inp=input_feat["attn.c_attn"],
                module2inspect=module.attn,
                kwargs=module_kwargs,
            )
        )

        # mlp
        layers.append(
            dict(
                prev_op=module.ln_2,
                layers=[module.mlp.w2, module.mlp.w1],
                inp=input_feat["mlp.w2"],
                module2inspect=module.mlp,
            )
        )

        # linear 2
        layers.append(
            dict(
                prev_op=module.mlp.w1,
                layers=[module.mlp.c_proj],
                inp=input_feat["mlp.c_proj"],
            )
        )

        return layers

def get_yi_layers(module: nn.Module, input_feat, module_kwargs):
        layers = []

        # attention input
        layers.append(dict(
            prev_op=module.ln1,
            layers=[module.self_attn.q_proj,
                    module.self_attn.k_proj, module.self_attn.v_proj],
            inp=input_feat['self_attn.q_proj'],
            module2inspect=module.self_attn, kwargs=module_kwargs,
        ))

        # attention out
        # Please refer to https://github.com/mit-han-lab/llm-/pull/67#issue-1850622696
        if module.self_attn.v_proj.weight.shape == module.self_attn.o_proj.weight.shape:
            layers.append(dict(
                prev_op=module.self_attn.v_proj,
                layers=[module.self_attn.o_proj],
                inp=input_feat['self_attn.o_proj'],
            ))
        
        # linear 1
        layers.append(dict(
            prev_op=module.ln2,
            layers=[module.mlp.gate_proj, module.mlp.up_proj],
            inp=input_feat['mlp.gate_proj'],
            module2inspect=module.mlp,
        ))

        # linear 2
        layers.append(dict(
            prev_op=module.mlp.up_proj,
            layers=[module.mlp.down_proj],
            inp=input_feat['mlp.down_proj'],
        ))

        return layers

true_sequential_layers_for_model = dict(
    AquilaForCausalLM=get_aquila_layers,
    BloomForCausalLM=get_bloom_layer,
    FalconForCausalLM=get_falcon_layers,
    GptBigCodeForCausalLM=get_bigcode_layers,
    GPTNeoXForCausalLM=get_neox_layers,
    GPTJForCausalLM=get_gptj_layers,
    MistralForCausalLM=get_mistray_layers,
    QwenForCausalLM=get_qwen_layers,
    YiForCausalLM=get_yi_layers,
)

def get_bloom_scaling(module: nn.Module):
    return dict(
            is_scalable=True,
            scale_name="mlp.gelu_impl",
            scale_layer=module.mlp.gelu_impl,
            scale_shape=module.mlp.dense_h_to_4h.out_features
    )
def get_falcon_scaling(module: nn.Module):
        return dict(
            is_scalable=True,
            scale_name="mlp.act",
            scale_layer=module.mlp.act,
            scale_shape=module.mlp.dense_h_to_4h.out_features
        )
def get_bigcode_scaling(module: nn.Module):
        return dict(
            is_scalable=True,
            scale_name="mlp.act",
            scale_layer=module.mlp.act,
            scale_shape=module.mlp.c_fc.out_features
        )
def get_neox_scaling(module: nn.Module):
    return dict(
        is_scalable=True,
        scale_name="mlp.act",
        scale_layer=module.mlp.act,
        scale_shape=module.mlp.dense_h_to_4h.out_features,
    )
def get_gptj_scaling(module: nn.Module):
        return dict(
            is_scalable=True,
            scale_name="mlp.act",
            scale_layer=module.mlp.act,
            scale_shape=module.mlp.fc_in.out_features
        )
def get_mpt_scaling(module: nn.Module):
        return dict(
            is_scalable=True,
            scale_name="ffn.act",
            scale_layer=module.ffn.act,
            scale_shape=module.ffn.up_proj.out_features
        )
_act_scales_map = dict(
      BloomForCausalLM=get_bloom_scaling,
      FalconForCausalLM=get_falcon_scaling,
      GptBigCodeForCausalLM=get_bigcode_scaling,
      GPTNeoXForCausalLM=get_neox_scaling,
      GPTJForCausalLM=get_gptj_scaling,
      MptForCausalLM=get_mpt_scaling,
)

def auto_detect_sequential_layers(module: nn.Module, input_feat, model_type, module_kwargs)->dict:
    assert model_type in true_sequential_layers_for_model, f"{model_type} is not support"
    return true_sequential_layers_for_model[model_type](module, input_feat, module_kwargs)

def auto_detect_scaling(module: nn.Module, model_type)->dict:
    if model_type not in _act_scales_map:return {}
    return _act_scales_map[model_type](module)