# refered from https://github.com/casper-hansen/AutoAWQ/tree/main/awq/models
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

def get_baichuan_layers(module: nn.Module, input_feat, module_kwargs):
    layers = []

    # attention input
    layers.append(dict(
        prev_op=module.input_layernorm,
        layers=[module.self_attn.W_pack],
        inp=input_feat['self_attn.W_pack'],
        module2inspect=module.self_attn, kwargs=module_kwargs,
    ))

    # # attention out
    # # Please refer to https://github.com/mit-han-lab/llm-awq/pull/67#issue-1850622696
    # if module.self_attn.v_proj.weight.shape == module.self_attn.o_proj.weight.shape:
    #     layers.append(dict(
    #         prev_op=module.self_attn.v_proj,
    #         layers=[module.self_attn.o_proj],
    #         inp=input_feat['self_attn.o_proj'],
    #     ))

    # attention out
    # Please refer to https://github.com/mit-han-lab/llm-awq/pull/67#issue-1850622696
    layers.append(dict(
        prev_op=module.self_attn.W_pack,
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

def get_cohere_layer(module: nn.Module, input_feat, module_kwargs):
    layers = []

    # attention input
    layers.append(
        dict(
            prev_op=module.input_layernorm,
            layers=[
                module.self_attn.q_proj,
                module.self_attn.k_proj,
                module.self_attn.v_proj,
                module.mlp.gate_proj,
                module.mlp.up_proj,
            ],
            inp=input_feat["self_attn.q_proj"],
            module2inspect=module,
            kwargs=module_kwargs,
        )
    )
    # attention out
    # Please refer to https://github.com/mit-han-lab/llm-awq/pull/67#issue-1850622696
    if module.self_attn.v_proj.weight.shape == module.self_attn.o_proj.weight.shape:
        layers.append(
            dict(
                prev_op=module.self_attn.v_proj,
                layers=[module.self_attn.o_proj],
                inp=input_feat["self_attn.o_proj"],
            )
        )

    # linear out
    layers.append(
        dict(
            prev_op=module.mlp.up_proj,
            layers=[module.mlp.down_proj],
            inp=input_feat["mlp.down_proj"],
        )
    )
    return layers

def get_deepseekv2_layers(module: nn.Module, input_feat, module_kwargs):
    layers = []
    if hasattr(module.self_attn, "q_proj"):
        # attention input
        layers.append(
            dict(
                prev_op=module.input_layernorm,
                layers=[
                    module.self_attn.q_proj,
                    module.self_attn.kv_a_proj_with_mqa,
                ],
                inp=input_feat["self_attn.q_proj"],
                module2inspect=module.self_attn,
                kwargs=module_kwargs,
            )
        )
    else:
        # attention input
        layers.append(
            dict(
                prev_op=module.input_layernorm,
                layers=[
                    module.self_attn.q_a_proj,
                    module.self_attn.kv_a_proj_with_mqa,
                ],
                inp=input_feat["self_attn.q_a_proj"],
                module2inspect=module.self_attn,
                kwargs=module_kwargs,
            )
        )
        layers.append(
            dict(
                prev_op=module.self_attn.q_a_layernorm,
                layers=[
                    module.self_attn.q_b_proj,
                ],
                inp=input_feat["self_attn.q_b_proj"],
            )
        )

    # kv layernorm
    layers.append(
        dict(
            prev_op=module.self_attn.kv_a_layernorm,
            layers=[
                module.self_attn.kv_b_proj,
            ],
            inp=input_feat["self_attn.kv_b_proj"],
        )
    )

    if hasattr(module.mlp, "gate"):
        # linear in
        layers.append(
            dict(
                prev_op=module.post_attention_layernorm,
                layers=[
                    w
                    for expert in module.mlp.experts
                    for w in [expert.gate_proj, expert.up_proj]
                ] + [module.mlp.shared_experts.gate_proj, module.mlp.shared_experts.up_proj],
                inp=input_feat["mlp"],
                module2inspect=module.mlp,
            )
        )

        # linear out
        for i, expert in enumerate(module.mlp.experts):
            layers.append(
                dict(
                    prev_op=expert.up_proj,
                    layers=[expert.down_proj],
                    inp=input_feat[f"mlp.experts.{i}.down_proj"],
                )
            )
        layers.append(
            dict(
                prev_op=module.mlp.shared_experts.up_proj,
                layers=[module.mlp.shared_experts.down_proj],
                inp=input_feat[f"mlp.shared_experts.down_proj"],
            )
        )
    else:
        # linear 1
        layers.append(
            dict(
                prev_op=module.post_attention_layernorm,
                layers=[module.mlp.gate_proj, module.mlp.up_proj],
                inp=input_feat["mlp.gate_proj"],
                module2inspect=module.mlp,
            )
        )

        # linear 2
        layers.append(
            dict(
                prev_op=module.mlp.up_proj,
                layers=[module.mlp.down_proj],
                inp=input_feat["mlp.down_proj"],
            )
        )

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

def get_gemma_layers(module: nn.Module, input_feat, module_kwargs):
    layers = []

    # attention input
    layers.append(
        dict(
            prev_op=module.input_layernorm,
            layers=[
                module.self_attn.q_proj,
                module.self_attn.k_proj,
                module.self_attn.v_proj,
            ],
            inp=input_feat["self_attn.q_proj"],
            module2inspect=module.self_attn,
            kwargs=module_kwargs,
        )
    )

    # attention out
    # Please refer to https://github.com/mit-han-lab/llm-awq/pull/67#issue-1850622696
    if module.self_attn.v_proj.weight.shape == module.self_attn.o_proj.weight.shape:
        layers.append(
            dict(
                prev_op=module.self_attn.v_proj,
                layers=[module.self_attn.o_proj],
                inp=input_feat["self_attn.o_proj"],
            )
        )

    # linear 1
    layers.append(
        dict(
            prev_op=module.post_attention_layernorm,
            layers=[module.mlp.gate_proj, module.mlp.up_proj],
            inp=input_feat["mlp.gate_proj"],
            module2inspect=module.mlp,
        )
    )

    # linear 2
    layers.append(
        dict(
            prev_op=module.mlp.up_proj,
            layers=[module.mlp.down_proj],
            inp=input_feat["mlp.down_proj"],
        )
    )

    return layers

def get_gemma2_layers(module: nn.Module, input_feat, module_kwargs):
    layers = []

    # attention input
    layers.append(
        dict(
            prev_op=module.input_layernorm,
            layers=[
                module.self_attn.q_proj,
                module.self_attn.k_proj,
                module.self_attn.v_proj,
            ],
            inp=input_feat["self_attn.q_proj"],
            module2inspect=module.self_attn,
            kwargs=module_kwargs,
        )
    )

    # attention out
    # Please refer to https://github.com/mit-han-lab/llm-awq/pull/67#issue-1850622696
    if module.self_attn.v_proj.weight.shape == module.self_attn.o_proj.weight.shape:
        layers.append(
            dict(
                prev_op=module.self_attn.v_proj,
                layers=[module.self_attn.o_proj],
                inp=input_feat["self_attn.o_proj"],
            )
        )

    layers.append(
        dict(
            prev_op=module.pre_feedforward_layernorm,
            layers=[module.mlp.gate_proj, module.mlp.up_proj],
            inp=input_feat["mlp.gate_proj"],
            module2inspect=module.mlp,
        )
    )

    layers.append(
        dict(
            prev_op=module.mlp.up_proj,
            layers=[module.mlp.down_proj],
            inp=input_feat["mlp.down_proj"],
        )
    )

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

def get_internlm2_layers(module, input_feat, module_kwargs):
    layers = []

    # attention input
    layers.append(
        dict(
            prev_op=module.attention_norm,
            layers=[
                module.attention.wqkv,
            ],
            inp=input_feat["attention.wqkv"],
            module2inspect=module.attention,
            kwargs=module_kwargs,
        )
    )

    # attention out
    layers.append(
        dict(
            prev_op=module.attention.wqkv,
            layers=[module.attention.wo],
            inp=input_feat["attention.wo"],
        )
    )

    # feed forward input
    layers.append(
        dict(
            prev_op=module.ffn_norm,
            layers=[
                module.feed_forward.w1,
                module.feed_forward.w3,
            ],
            inp=input_feat["feed_forward.w1"],
            module2inspect=module.feed_forward,
            kwargs=module_kwargs,
        )
    )

    # feed forward output
    layers.append(
        dict(
            prev_op=module.feed_forward.w1,
            layers=[module.feed_forward.w2],
            inp=input_feat["feed_forward.w2"],
        )
    )

    return layers

def get_llama_layers(module: nn.Module, input_feat, module_kwargs):
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
    # Please refer to https://github.com/mit-han-lab/llm-awq/pull/67#issue-1850622696
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

def get_llava_layers(module: nn.Module, input_feat, module_kwargs):
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
    # Please refer to https://github.com/mit-han-lab/llm-awq/pull/67#issue-1850622696
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

def get_llavanext_layers(module: nn.Module, input_feat, module_kwargs):
    layers = []

    # attention input
    layers.append(
        dict(
            prev_op=module.input_layernorm,
            layers=[
                module.self_attn.q_proj,
                module.self_attn.k_proj,
                module.self_attn.v_proj,
            ],
            inp=input_feat["self_attn.q_proj"],
            module2inspect=module.self_attn,
            kwargs=module_kwargs,
        )
    )

    # attention out
    # Please refer to https://github.com/mit-han-lab/llm-awq/pull/67#issue-1850622696
    if module.self_attn.v_proj.weight.shape == module.self_attn.o_proj.weight.shape:
        layers.append(
            dict(
                prev_op=module.self_attn.v_proj,
                layers=[module.self_attn.o_proj],
                inp=input_feat["self_attn.o_proj"],
            )
        )

    # linear 1
    layers.append(
        dict(
            prev_op=module.post_attention_layernorm,
            layers=[module.mlp.gate_proj, module.mlp.up_proj],
            inp=input_feat["mlp.gate_proj"],
            module2inspect=module.mlp,
        )
    )

    # linear 2
    layers.append(
        dict(
            prev_op=module.mlp.up_proj,
            layers=[module.mlp.down_proj],
            inp=input_feat["mlp.down_proj"],
        )
    )

    return layers

def get_minicpm_layers(module, input_feat, module_kwargs):
    layers = []

    

    # # mlp
    layers.append(
        dict(
            prev_op=module.input_layernorm,
            layers=[
                module.self_attn.q_proj,
                module.self_attn.k_proj,
                module.self_attn.v_proj,
            ],
            inp=input_feat["self_attn.q_proj"],
            module2inspect=module.self_attn,
            kwargs=module_kwargs,
        )
    )

    # linear 2
    layers.append(
        dict(
            prev_op=module.mlp.up_proj,
            layers=[module.mlp.down_proj],
            inp=input_feat["mlp.down_proj"],
        )
    )

    layers.append(
        dict(
            prev_op=module.post_attention_layernorm,
            layers=[module.mlp.gate_proj,module.mlp.up_proj],
            inp=input_feat["mlp.gate_proj"],
            module2inspect=module.mlp
        )
    )

    return layers

def get_mistral_layers(module: nn.Module, input_feat, module_kwargs):
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

def get_mixtral_layers(module: nn.Module, input_feat, module_kwargs):
    #Workaround for Mixtral, exclude layers which are not quantized
    input_feat.pop('block_sparse_moe.gate', None)
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
    if module.self_attn.v_proj.weight.shape == module.self_attn.o_proj.weight.shape:
        layers.append(dict(
            prev_op=module.self_attn.v_proj,
            layers=[module.self_attn.o_proj],
            inp=input_feat['self_attn.o_proj'],
        ))

    # linear in
    layers.append(dict(
        prev_op=module.post_attention_layernorm,
        layers=[
            w for expert in module.block_sparse_moe.experts
            for w in [expert.w1, expert.w3]
        ],
        inp=input_feat['block_sparse_moe'],
        module2inspect=module.block_sparse_moe,
    ))

    # linear out
    for i, expert in enumerate(module.block_sparse_moe.experts):
        layers.append(dict(
            prev_op=expert.w3,
            layers=[expert.w2],
            inp=input_feat[f'block_sparse_moe.experts.{i}.w2'],
        ))

    return layers

def get_mpt_layers(module: nn.Module, input_feat, module_kwargs):
    layers = []

    if module_kwargs.get("output_attentions") is not None:
        module_kwargs.pop("output_attentions")

    # attention input
    layers.append(dict(
        prev_op=module.norm_1,
        layers=[module.attn.Wqkv],
        inp=input_feat['attn.Wqkv'],
        module2inspect=module.attn,
        kwargs=module_kwargs
    ))

    # attention output
    layers.append(dict(
        prev_op=module.attn.Wqkv,
        layers=[module.attn.out_proj],
        inp=input_feat['attn.out_proj']
    ))

    # linear 1
    layers.append(dict(
        prev_op=module.norm_2,
        layers=[module.ffn.up_proj],
        inp=input_feat['ffn.up_proj'],
        module2inspect=module.ffn
    ))

    # linear 2
    layers.append(dict(
        prev_op=module.ffn.act,
        layers=[module.ffn.down_proj],
        inp=input_feat['ffn.down_proj']
    ))

    return layers

def get_opt_layers(module: nn.Module, input_feat, module_kwargs):
    layers = []

    # attention input
    layers.append(dict(
        prev_op=module.self_attn_layer_norm,
        layers=[
            module.self_attn.q_proj,
            module.self_attn.k_proj, module.self_attn.v_proj],
        inp=input_feat['self_attn.q_proj'],
        module2inspect=module.self_attn,
        kwargs=module_kwargs,
    ))

    # attention out
    layers.append(dict(
        prev_op=module.self_attn.v_proj,
        layers=[module.self_attn.out_proj],
        inp=input_feat['self_attn.out_proj'],
    ))

    # linear 1
    layers.append(dict(
        prev_op=module.final_layer_norm,
        layers=[module.fc1],
        inp=input_feat['fc1'],
    ))

    # linear 2
    layers.append(dict(
        prev_op=module.fc1,
        layers=[module.fc2],
        inp=input_feat['fc2'],
    ))

    return layers

def get_phi3_layers(module: nn.Module, input_feat, module_kwargs):
    layers = []

    # attention input
    layers.append(
        dict(
            prev_op=module.input_layernorm,
            layers=[module.self_attn.qkv_proj],
            inp=input_feat["self_attn.qkv_proj"],
            module2inspect=module.self_attn,
            kwargs=module_kwargs,
        )
    )

    # attention out
    layers.append(
        dict(
            prev_op=module.self_attn.qkv_proj,
            layers=[module.self_attn.o_proj],
            inp=input_feat["self_attn.o_proj"],
        )
    )

    # linear 1
    layers.append(
        dict(
            prev_op=module.post_attention_layernorm,
            layers=[module.mlp.gate_up_proj],
            inp=input_feat["mlp.gate_up_proj"],
            module2inspect=module.mlp,
        )
    )

    # linear 2
    layers.append(
        dict(
            prev_op=module.mlp.gate_up_proj,
            layers=[module.mlp.down_proj],
            inp=input_feat["mlp.down_proj"],
        )
    )

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

def get_qwen2_layers(module: nn.Module, input_feat, module_kwargs):
    layers = []

    # attention input
    layers.append(
        dict(
            prev_op=module.input_layernorm,
            layers=[
                module.self_attn.q_proj,
                module.self_attn.k_proj,
                module.self_attn.v_proj,
            ],
            inp=input_feat["self_attn.q_proj"],
            module2inspect=module.self_attn,
            kwargs=module_kwargs,
        )
    )

    # attention out
    # Please refer to https://github.com/mit-han-lab/llm-awq/pull/67#issue-1850622696
    if module.self_attn.v_proj.weight.shape == module.self_attn.o_proj.weight.shape:
        layers.append(
            dict(
                prev_op=module.self_attn.v_proj,
                layers=[module.self_attn.o_proj],
                inp=input_feat["self_attn.o_proj"],
            )
        )

    # linear 1
    layers.append(
        dict(
            prev_op=module.post_attention_layernorm,
            layers=[module.mlp.gate_proj, module.mlp.up_proj],
            inp=input_feat["mlp.gate_proj"],
            module2inspect=module.mlp,
        )
    )

    # linear 2
    layers.append(
        dict(
            prev_op=module.mlp.up_proj,
            layers=[module.mlp.down_proj],
            inp=input_feat["mlp.down_proj"],
        )
    )

    return layers

def get_stablelm_layers(
        module: nn.Module, input_feat, module_kwargs
    ):
    layers = []

    # attention input
    layers.append(
        dict(
            prev_op=module.input_layernorm,
            layers=[
                module.self_attn.q_proj,
                module.self_attn.k_proj,
                module.self_attn.v_proj,
            ],
            inp=input_feat["self_attn.q_proj"],
            module2inspect=module.self_attn,
            kwargs=module_kwargs,
        )
    )

    # attention out
    # Please refer to https://github.com/mit-han-lab/llm-awq/pull/67#issue-1850622696
    if module.self_attn.v_proj.weight.shape == module.self_attn.o_proj.weight.shape:
        layers.append(
            dict(
                prev_op=module.self_attn.v_proj,
                layers=[module.self_attn.o_proj],
                inp=input_feat["self_attn.o_proj"],
            )
        )

    # linear 1
    layers.append(
        dict(
            prev_op=module.post_attention_layernorm,
            layers=[module.mlp.gate_proj, module.mlp.up_proj],
            inp=input_feat["mlp.gate_proj"],
            module2inspect=module.mlp,
        )
    )

    # linear 2
    layers.append(
        dict(
            prev_op=module.mlp.up_proj,
            layers=[module.mlp.down_proj],
            inp=input_feat["mlp.down_proj"],
        )
    )

    return layers

def get_starcoder2_layers(module: nn.Module, input_feat, module_kwargs):
    layers = []

    # attention input
    layers.append(
        dict(
            prev_op=module.input_layernorm,
            layers=[
                module.self_attn.q_proj,
                module.self_attn.k_proj,
                module.self_attn.v_proj,
            ],
            inp=input_feat["self_attn.q_proj"],
            module2inspect=module.self_attn,
            kwargs=module_kwargs,
        )
    )

    # attention out
    if module.self_attn.v_proj.weight.shape == module.self_attn.o_proj.weight.shape:
        layers.append(
            dict(
                prev_op=module.self_attn.v_proj,
                layers=[module.self_attn.o_proj],
                inp=input_feat["self_attn.o_proj"],
            )
        )

    # linear 1
    layers.append(
        dict(
            prev_op=module.post_attention_layernorm,
            layers=[module.mlp.c_fc],
            inp=input_feat["mlp.c_fc"],
            module2inspect=module.mlp,
        )
    )

    # linear 2
    layers.append(
        dict(
            prev_op=module.mlp.act,
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
    BaichuanForCausalLM=get_baichuan_layers,
    BloomForCausalLM=get_bloom_layer,
    CohereForCausalLM=get_cohere_layer,
    DeepseekV2ForCausalLM=get_deepseekv2_layers,
    FalconForCausalLM=get_falcon_layers,
    GemmaForCausalLM=get_gemma_layers,
    Gemma2ForCausalLM=get_gemma2_layers,
    GptBigCodeForCausalLM=get_bigcode_layers,
    GPTNeoXForCausalLM=get_neox_layers,
    GPTJForCausalLM=get_gptj_layers,
    InternLM2ForCausalLM=get_internlm2_layers,
    LlamaForCausalLM=get_llama_layers,
    LlavaForCausalLM =get_llava_layers,
    MiniCPMForCausalLM=get_minicpm_layers,
    MistralForCausalLM=get_mistral_layers,
    MixtralForCausalLM =get_mixtral_layers,
    MptForCausalLM =get_mpt_layers,
    OPTForCausalLM =get_opt_layers,
    Phi3ForCausalLM=get_phi3_layers,
    QwenForCausalLM=get_qwen_layers,
    Qwen2ForCausalLM=get_qwen2_layers,
    StableLmForCausalLM=get_stablelm_layers,
    Starcoder2ForCausalLM=get_starcoder2_layers,
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
def get_internlm2_scaling(module):
    return dict(
        is_scalable=True,
        scale_name="feed_forward.w2",
        scale_layer=module.feed_forward.w2,
        scale_shape=module.feed_forward.w2.out_features,
    )
def get_mpt_scaling(module: nn.Module):
    return dict(
        is_scalable=True,
        scale_name="ffn.act",
        scale_layer=module.ffn.act,
        scale_shape=module.ffn.up_proj.out_features
    )
def get_starcoder2_scaling(module: nn.Module):
    return dict(
        is_scalable=True,
        scale_name="mlp.act",
        scale_layer=module.mlp.act,
        scale_shape=module.mlp.c_fc.out_features,
    )

_act_scales_map = dict(
      BloomForCausalLM=get_bloom_scaling,
      FalconForCausalLM=get_falcon_scaling,
      GptBigCodeForCausalLM=get_bigcode_scaling,
      GPTNeoXForCausalLM=get_neox_scaling,
      GPTJForCausalLM=get_gptj_scaling,
      InternLM2ForCausalLM=get_internlm2_scaling,
      MptForCausalLM=get_mpt_scaling,
      Starcoder2ForCausalLM=get_starcoder2_scaling,
)

def auto_detect_sequential_layers(module: nn.Module, input_feat, model_type, module_kwargs)->dict:
    assert model_type in true_sequential_layers_for_model, f"{model_type} is not support"
    return true_sequential_layers_for_model[model_type](module, input_feat, module_kwargs)

def auto_detect_scaling(module: nn.Module, model_type)->dict:
    if model_type not in _act_scales_map: return dict(is_scalable=False)
    return _act_scales_map[model_type](module)