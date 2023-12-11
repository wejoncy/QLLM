import torch
import torch.nn as nn

true_sequential_layers_for_model = dict(
    RWForCausalLM=[
        ["self_attention.query_key_value"],
        ["self_attention.dense"],
        ["mlp.dense_h_to_4h"],
        ["mlp.dense_4h_to_h"]
    ],
    BaiChuanForCausalLM=[
        ["self_attn.W_pack"],
        ["self_attn.o_proj"],
        ["mlp.up_proj", "mlp.gate_proj"],
        ["mlp.down_proj"]
    ],
    BloomForCausalLM=[
        ["self_attention.query_key_value"],
        ["self_attention.dense"],
        ["mlp.dense_h_to_4h"],
        ["mlp.dense_4h_to_h"]
    ],
    CodeGenForCausalLM=[
        ["attn.qkv_proj"],
        ["attn.out_proj"],
        ["mlp.fc_in"],
        ["mlp.fc_out"]
    ],    GPT2ForCausalLM=[
        ["attn.qkv_proj"],
        ["attn.out_proj"],
        ["mlp.fc_in"],
        ["mlp.fc_out"]
    ],    GPTBigCodeForCausalLM=[
        ["attn.c_attn"],
        ["attn.c_proj"],
        ["mlp.c_fc"],
        ["mlp.c_proj"]
    ],    GPTNeoXForCausalLM=[
        ["attention.query_key_value"],
        ["attention.dense"],
        ["mlp.dense_h_to_4h"],
        ["mlp.dense_4h_to_h"]
    ],    GPTJForCausalLM=[
        ["attn.k_proj", "attn.v_proj", "attn.q_proj"],
        ["attn.out_proj"],
        ["mlp.fc_in"],
        ["mlp.fc_out"]
    ],    InternLMForCausalLM=[
        ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
        ["self_attn.o_proj"],
        ["mlp.up_proj", "mlp.gate_proj"],
        ["mlp.down_proj"],
    ],    LlamaForCausalLM=[
        ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
        ["self_attn.o_proj"],
        ["mlp.up_proj", "mlp.gate_proj"],
        ["mlp.down_proj"]
    ], MistralForCausalLM=[
        ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
        ["self_attn.o_proj"],
        ["mlp.up_proj", "mlp.gate_proj"],
        ["mlp.down_proj"],
    ], MOSSForCausalLM=[
        ["attn.qkv_proj"],
        ["attn.out_proj"],
        ["mlp.fc_in"],
        ["mlp.fc_out"]
    ], OPTForCausalLM=[
        ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
        ["self_attn.out_proj"],
        ["fc1"],
        ["fc2"]
    ], QwenForCausalLM=[
        ["attn.c_attn"],
        ["attn.c_proj"],
        ["mlp.w1", "mlp.w2"],
        ["mlp.c_proj"]
    ], XverseForCausalLM=[
        ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
        ["self_attn.o_proj"],
        ["mlp.up_proj", "mlp.gate_proj"],
        ["mlp.down_proj"]
    ], YiForCausalLM=[
        ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
        ["self_attn.o_proj"],
        ["mlp.up_proj", "mlp.gate_proj"],
        ["mlp.down_proj"]
    ],
)


def auto_detect_sequential_layers(flatten_layers, model_type):
    if model_type in true_sequential_layers_for_model:
        return true_sequential_layers_for_model[model_type]
    top_layers = []
    layers = [flatten_layers[0][0]]
    for i in range(1, len(flatten_layers[0])):
        if flatten_layers[0][i].split('.')[0] == layers[-1].split('.')[0]:
            layers.append(flatten_layers[0][i])
        else:
            top_layers.append(layers)
            layers = [flatten_layers[0][i]]
    top_layers.append(layers)

    # filter out o_projection
    top_layers.insert(1, [])
    top_layers[1].append(top_layers[0][-1])
    top_layers[0].pop(-1)
    top_layers.append([])
    top_layers[-1].append(top_layers[-2][-1])
    top_layers[-2].pop(-1)
    return top_layers