# QLLM
<p align="center">
    <a href="https://colab.research.google.com/github/wejoncy/QLLM/blob/main/qllm_colab.ipynb">
        <img alt="Open In Colab" src="https://colab.research.google.com/assets/colab-badge.svg">
    </a>
    <a href="https://github.com/wejoncy/QLLM/releases">
        <img alt="GitHub - Releases" src="https://img.shields.io/github/v/release/wejoncy/QLLM.svg">
    </a>
    <a href="https://pypi.org/project/qllm/">
        <img alt="PyPI - Downloads" src="https://static.pepy.tech/badge/qllm/month">
    </a>
</p>

KeyWords **Quantization**, **GPTQ,AWQ, HQQ**, **ONNX, ONNXRuntime**, **VLLM**

<font size=5>
<center>Quantize all LLMs in HuggingFace/Transformers with GPTQ/AWQ/HQQ in mixed bits(2-8bit), and export to onnx model</center>
</font>
<br><br>
QLLM is a out-of-box quantization toolbox for large language models, It is designed to be a auto-quantization framework which takes layer by layer for any LLMs. It can also be used to export quantized model to onnx with only one args `--export_onnx ./onnx_model`, and inference with onnxruntime.
Besides, model quantized by different quantization method (GPTQ/AWQ/HQQ) can be loaded from huggingface/transformers and transfor to each other without extra effort. 

We alread supported 
- [x] GPTQ quantization 
- [x] AWQ quantization
- [x] HQQ quantization


Features:
- [x] GPTQ supports all LLM models in huggingface/transformers, it will automatically detect the model type and quantize it.
- [x] We support to quantize model by 2-8 bits, and support to quantize model with different quantization bits for different layers.
- [x] Auto promoting bits/group-size for better accuracy
- [x] Export to ONNX model, Running by OnnxRuntime 

*Latest News* 🔥
- [2024/03] ONNX Models export API
- [2024/01] Support [HQQ](https://github.com/mobiusml/hqq) algorithm
- [2023/12] The first PyPi package released 

## Installation
Easy to install qllm from PyPi [cu121]

`pip install qllm`


Install from release package, CUDA-118/121 is supported.
[py38, py39, py310] https://github.com/wejoncy/QLLM/releases

Build from Source

**Please set ENV EXCLUDE_EXTENTION_FOR_FAST_BUILD=1 for fast build**

If you are using CUDA-121
```
pip install git+https://github.com/wejoncy/QLLM.git --no-build-isolation
```
OR CUDA-118/117
```
git clone https://github.com/wejoncy/QLLM.git
cd QLLM
python setup.py install
```

# How to use it

## Quantize llama2
```bash
#  Quantize and Save compressed model, method can be one of [gptq/awq/hqq]
python -m qllm --model=meta-llama/Llama-2-7b-hf --method=gptq --nsamples=64 --wbits=4 --groupsize=128 --save ./Llama-2-7b-4bit
python -m qllm --model=meta-llama/Llama-2-7b-hf --method=awq --dataset=pileval --nsamples=16 --wbits=4 --groupsize=128 --save ./Llama-2-7b-4bit
python -m qllm --model=meta-llama/Llama-2-7b-hf --method=hqq --wbits=4 --groupsize=128 --save ./Llama-2-7b-4bit
```

## Convert to onnx model
use `--export_onnx ./onnx_model` to export and save onnx model
```
python -m qllm --model  meta-llama/Llama-2-7b-chat-hf  --method=gptq  --dataset=pileval --nsamples=16  --save ./Llama-2-7b-chat-hf_awq_q4/ --export_onnx ./Llama-2-7b-chat-hf_awq_q4_onnx/
```
or you can convert a existing model in HF Hub
```
python -m qllm --load TheBloke/Llama-2-7B-Chat-AWQ --export_onnx=./onnx
python -m qllm --load TheBloke/Llama-2-7B-Chat-GPTQ --export_onnx=./onnx
```

## (NEW) Quantize model with mix bits/groupsize for higher precision (PPL)
```bash
#  Quantize and Save compressed model
python -m qllm --model=meta-llama/Llama-2-7b-hf --method=gptq --save ./Llama-2-7b-4bit --allow_mix_bits --true-sequential
```
### NOTE:
1. only support GPTQ
2. allow_mix_bits option refered from gptq-for-llama, QLLM makes it easier to use and flexible
3. wjat different with gptq-for-llama is we grow bit by one instead of times 2.
4. all configurations will be saved/load automaticlly instead of quant-table which used by gptq-for-llama.
5. if --allow_mix_bits is enabled, The saved model is not compatible with vLLM for now.

## Quantize model for vLLM 
Due to the zereos diff, we need to set a env variable if you set pack_mode to GPTQ whenver the method is awq or gptq
```bash
COMPATIBLE_WITH_AUTOGPTQ=1 python -m qllm --model=meta-llama/Llama-2-7b-hf --method=gptq --save ./Llama-2-7b-4bit --pack_mode=GPTQ
```
If you use GEMM pack_mode, then you don't have to set the var
```bash
python -m qllm --model=meta-llama/Llama-2-7b-hf --method=gptq --save ./Llama-2-7b-4bit --pack_mode=GEMM
```

```bash
python -m qllm --model=meta-llama/Llama-2-7b-hf --method=awq --save ./Llama-2-7b-4bit --pack_mode=GEMM
```
## Conversion among AWQ, GPTQ and MarLin
```bash
python -m qllm --load TheBloke/Llama-2-7B-Chat-AWQ --eval --save ./Llama-2-7b-chat-hf_gptq_q4/ --pack_mode=GPTQ
```
Or you can use `--pack_mode=AWQ` to convert GPTQ to AWQ.
```bash
python -m qllm --load TheBloke/Llama-2-7B-Chat-GPTQ --eval --save ./Llama-2-7b-chat-hf_awq_q4/ --pack_mode=GEMM
```
Or you can use `--pack_mode=MARLIN` to convert GPTQ to Marlin.
```bash
python -m qllm --load TheBloke/Llama-2-7B-Chat-GPTQ --eval --save ./Llama-2-7b-chat-hf_marlin_q4/ --pack_mode=MARLIN
```
Or you can use `--pack_mode=MARLIN` to convert AWQ to Marlin.
```bash
python -m qllm --load TheBloke/Llama-2-7B-Chat-AWQ --eval --save ./Llama-2-7b-chat-hf_marlin_q4/ --pack_mode=MARLIN
```

### Note:
Not all cases are supported, for example,
1)  if you quantized model with different quantization bits for different layers, you can't convert it to AWQ.
2)  if GPTQ model is quantized with `--allow_mix_bits` option, you can't convert it to AWQ.
3)  if GPTQ model is quantized with `--act_order` option, you can't convert it to AWQ.


## model inference with the saved model
```bash
python -m qllm --load ./Llama-2-7b-4bit --eval
```

## model inference with ORT
you may want to use [genai](https://github.com/microsoft/onnxruntime-genai) to do generation with ORT.
```python
import onnxruntime
from transformers import AutoTokenizer
onnx_path_str = './Llama-2-7b-4bit-onnx'

tokenizer = AutoTokenizer.from_pretrained(onnx_path_str, use_fast=True)
sample_inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
onnx_model_path = onnx_path_str+'/decoder_merged.onnx'
session = onnxruntime.InferenceSession(onnx_model_path, providers=['CUDAExecutionProvider'])
mask = np.ones(sample_inputs[0].shape, dtype=np.int64) if sample_inputs[1] is None else sample_inputs[1].cpu().numpy()
num_layers = model.config.num_hidden_layers
inputs = {'input_ids': sample_inputs[0].cpu().numpy(), 
          'attention_mask': mask, 
          'position_ids': np.arrange(0,sample_inputs[0], dtype=np.int64),
          'use_cache_branch': np.array([0], dtype=np.bool_)}
for i in range(num_layers):
    inputs[f'past_key_values.{i}.key'] = np.zeros((1, 32, 32, 128), dtype=np.float16)
    inputs[f'past_key_values.{i}.value'] = np.zeros((1, 32, 32, 128), dtype=np.float16)
outputs = session(None, inputs)
```

## Load quantized model from hugingface/transformers
```bash
python -m qllm --load TheBloke/Llama-2-7B-Chat-AWQ --eval
python -m qllm --load TheBloke/Llama-2-7B-Chat-GPTQ --eval
python -m qllm --load TheBloke/Mixtral-8x7B-v0.1-GPTQ  --use_plugin
```

## start a chatbot
you may need to install fschat and accelerate with pip
```bash
pip install fschat accelerate
```
use `--use_plugin` to enable a chatbot plugin

```
python -m qllm --model  meta-llama/Llama-2-7b-chat-hf  --method=awq  --dataset=pileval --nsamples=16  --use_plugin --save ./Llama-2-7b-chat-hf_awq_q4/

or 
python -m qllm --model  meta-llama/Llama-2-7b-chat-hf  --method=gptq  --dataset=pileval --nsamples=16  --use_plugin --save ./Llama-2-7b-chat-hf_gptq_q4/
```

## use QLLM with API
```python
from qllm import AutoModelQuantization

quantizer = AutoModelQuantization()
q_model = quantizer.api_quantize(model_or_model_path='meta-llama/Llama-2-7b-hf', method='gptq', wbits=4, groupsize=128)
```
OR 
```python
from qllm import AutoModelQuantization
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", use_fast=True, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", torch_dtype=torch.float16)
q_model = quantizer.api_quantize(model_or_model_path=q_model, method='gptq', wbits=4, groupsize=128)
```

## For some users has transformers connect issues.
Please set environment with PROXY_PORT=your http proxy port

PowerShell
`$env:PROXY_PORT=1080`

Bash
`export PROXY_PORT=1080`

windows cmd
`set PROXY_PORT=1080`

# Acknowledgements
[GPTQ](https://github.com/IST-DASLab/gptq)

[GPTQ-triton](https://github.com/fpgaminer/GPTQ-triton)

[AutoGPTQ](https://github.com/PanQiWei/AutoGPTQ)

[llm-awq](https://github.com/mit-han-lab/llm-awq)

[AutoAWQ](https://github.com/casper-hansen/AutoAWQ).

[HQQ](https://github.com/mobiusml/hqq)
