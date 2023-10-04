# QLLM
![qllm](https://github.com/wejoncy/QLLM/assets/fb201d9c-f889-4504-9ef5-ac77ec1cd8e2.jpg)

Supporting GPTQ quantization method.
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

# Acknowledgements
This code is based on [GPTQ](https://github.com/IST-DASLab/gptq)

Thanks to Meta AI for releasing [LLaMA](https://arxiv.org/abs/2302.13971), a powerful LLM.

Triton GPTQ kernel code is based on [GPTQ-triton](https://github.com/fpgaminer/GPTQ-triton)
