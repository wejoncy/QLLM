# QLLM
<img src="https://github.com/wejoncy/QLLM/blob/main/assets/fb201d9c-f889-4504-9ef5-ac77ec1cd8e2.jpg?raw=true" width="210">

QLLM is a out-of-box quantization toolbox for large language models, It didn't limit to a specific model, and designed to be auto-quantization layer by layer for any LLMs. It can also be used to export quantized model to onnx with only one args `--export_onnx ./onnx_model`, and inference with onnxruntime.
Besides, model quantized by different quantization method (GPTQ/AWQ) can be loaded from huggingface/transformers and transfor to each other without extra effort. 

We alread supported 
- [x] GPTQ quantization 
- [x] AWQ quantization


Features:
- [x] GPTQ supports all LLM models in huggingface/transformers, it will automatically detect the model type and quantize it.
- [x] for GPTQ, we support to quantize model by 2-8 bits, and support to quantize model with different quantization bits for different layers.
- [x] for AWQ, we support only those models in llm-awq/auto-awq for now.
- [x] we support to load  model which quantized by AutoGPTQ and AutoAWQ.
- [x] we only support **Nvidia-GPU** platform for now,
- [ ] we will consider support **AMD-GPU**.

## Installation
```
pip install git+https://github.com/wejoncy/QLLM.git
```
## Dependencies

* `torch`: tested on v2.0.0+cu117
* `transformers`: tested on v4.28.0.dev0
* `datasets`: tested on v2.10.1
* `safetensors`: tested on v0.3.0
* `onnxruntime`: tested on v1.16.1
* `onnx`


# How to use it

## Quantize llama2
```bash
#  Quantize and Save compressed model
CUDA_VISIBLE_DEVICES=0 python -m qllm.run --model=meta-llama/Llama-2-7b-hf --method=gptq --save ./Llama-2-7b-4bit
```

## (NEW) Quantize model with mix bits/groupsize for higher precision (PPL)
```bash
#  Quantize and Save compressed model
python -m qllm.run --model=meta-llama/Llama-2-7b-hf --method=gptq --save ./Llama-2-7b-4bit --observe --true-sequential
```
### NOTE:
1. only support GPTQ
2. observe option refered from gptq-for-llama, QLLM makes it easier to use and flexible
3. wjat different with gptq-for-llama is we grow bit by one instead of times 2.
4. all configurations will be saved/load automaticlly instead of quant-table which used by gptq-for-llama.
5. if --observe is enabled, The saved model is not compatible with vLLM for now.


## Convert to onnx model
use `--export_onnx ./onnx_model` to export and save onnx model
```
python -m qllm.run --model  meta-llama/Llama-2-7b-chat-hf  --method=gptq  --dataset=pileval --nsamples=16  --save ./Llama-2-7b-chat-hf_awq_q4/ --export_onnx ./Llama-2-7b-chat-hf_awq_q4_onnx/
```

## model inference with the saved model
```bash
CUDA_VISIBLE_DEVICES=0 python -m qllm.run --load ./Llama-2-7b-4bit --eval
```

## model inference with ORT
```python
import onnxruntime
from transformers import AutoTokenizer
onnx_path_str = './Llama-2-7b-4bit-onnx'

tokenizer = AutoTokenizer.from_pretrained(onnx_path_str, use_fast=True)
sample_inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
onnx_model_path = onnx_path_str+'/model_one_for_all.onnx'
session = onnxruntime.InferenceSession(onnx_model_path, providers=['CUDAExecutionProvider'])
mask = np.ones(sample_inputs[0].shape, dtype=np.int64) if sample_inputs[1] is None else sample_inputs[1].cpu().numpy()
num_layers = model.config.num_hidden_layers
inputs = {'input_ids': sample_inputs[0].cpu().numpy(), 'attention_mask': mask, 'use_cache_branch': np.array([0], dtype=np.bool_)}
for i in range(num_layers):
    inputs[f'present_key.{i}'] = np.zeros((1, 32, 32, 128), dtype=np.float16)
    inputs[f'present_values.{i}'] = np.zeros((1, 32, 32, 128), dtype=np.float16)
outputs = session.run(None, inputs)
```

## Load quantized model from hugingface/transformers
```bash
CUDA_VISIBLE_DEVICES=0 python -m qllm.run --load TheBloke/Llama-2-7B-Chat-AWQ --eval
CUDA_VISIBLE_DEVICES=0 python -m qllm.run --load TheBloke/Llama-2-7B-Chat-GPTQ --eval
```

## start a chatbot
use `--use_plugin` to enable a chatbot plugin

```
python -m qllm.run --model  meta-llama/Llama-2-7b-chat-hf  --method=awq  --dataset=pileval --nsamples=16  --use_plugin --save ./Llama-2-7b-chat-hf_awq_q4/

or 
python -m qllm.run --model  meta-llama/Llama-2-7b-chat-hf  --method=gptq  --dataset=pileval --nsamples=16  --use_plugin --save ./Llama-2-7b-chat-hf_gptq_q4/
```


# Acknowledgements
This code is based on [GPTQ](https://github.com/IST-DASLab/gptq)

Triton GPTQ kernel code is based on [GPTQ-triton](https://github.com/fpgaminer/GPTQ-triton)

Thanks to [AutoGPTQ](https://github.com/PanQiWei/AutoGPTQ)

Thanks to [llm-awq](https://github.com/mit-han-lab/llm-awq) and [AutoAWQ](https://github.com/casper-hansen/AutoAWQ) for releasing AWQ quantization method.
