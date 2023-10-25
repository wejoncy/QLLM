# QLLM
<img src="https://github.com/wejoncy/QLLM/blob/main/assets/fb201d9c-f889-4504-9ef5-ac77ec1cd8e2.jpg?raw=true" width="420">

We alread supported 
[x] GPTQ quantization 
[x] AWQ quantization. refered to [llm-awq](https://github.com/mit-han-lab/llm-awq) and [AutoAWQ](https://github.com/casper-hansen/AutoAWQ)

We support 2-8 bits quantization and conresspoing kernels on **Nvidia-GPU**, we will consider support **AMD-GPU** too.

## Installation
```
pip install git+https://github.com/wejoncy/QLLM.git
```
## Dependencies

* `torch`: tested on v2.0.0+cu117
* `transformers`: tested on v4.28.0.dev0
* `datasets`: tested on v2.10.1
* `safetensors`: tested on v0.3.0
* `onnxruntime`: 
* `onnx`

# Language Generation

```
# Save compressed model
CUDA_VISIBLE_DEVICES=0 python -m qllm.model_quantization_base --model=meta-llama/Llama-2-7b-hf --save ./Llama-2-7b-4bit

# convert to onnx model and save torch model
python -m qllm.model_quantization_base --model=meta-llama/Llama-2-7b-hf --save ./Llama-2-7b-4bit --onnx ././Llama-2-7b-4bit-onnx

# model inference with the saved model
CUDA_VISIBLE_DEVICES=0 python -m qllm.model_quantization_base --load ./Llama-2-7b-4bit --eval

# model inference with ORT
TO be DONE
```

# start a chatbot
use `--use_plugin` to enable a chatbot plugin

```
python -m qllm.model_quantization_base --model  meta-llama/Llama-2-7b-chat-hf/  --method=awq  --dataset=pileval --nsamples=16  --use_plugin --save ./Llama-2-7b-chat-hf_awq_q4/

or 
python -m qllm.model_quantization_base --model  meta-llama/Llama-2-7b-chat-hf/  --method=gptq  --dataset=pileval --nsamples=16  --use_plugin --save ./Llama-2-7b-chat-hf_gptq_q4/
```

# Convert to onnx model
use `--export_onnx ./onnx_model` to export and save onnx model
```
python -m qllm.model_quantization_base --model  meta-llama/Llama-2-7b-chat-hf/  --method=gptq  --dataset=pileval --nsamples=16  --save ./Llama-2-7b-chat-hf_awq_q4/ --export_onnx ./Llama-2-7b-chat-hf_awq_q4_onnx/
```

# Acknowledgements
This code is based on [GPTQ](https://github.com/IST-DASLab/gptq)

Thanks to Meta AI for releasing [LLaMA](https://arxiv.org/abs/2302.13971), a powerful LLM.

Triton GPTQ kernel code is based on [GPTQ-triton](https://github.com/fpgaminer/GPTQ-triton)

Thanks to [GPTQ-for-LLaMa](https://github.com/qwopqwop200/GPTQ-for-LLaMa)
