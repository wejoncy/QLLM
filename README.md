# QLLM
![qllm](https://github.com/wejoncy/QLLM/assets/9417365/776a8ebd-43ea-4657-853a-2ebcfd4bed34)

Supporting GPTQ quantization method.
We support 2-8 bits quantization and conresspoing kernels on **Nvidia-GPU**, we will consider support **AMD-GPU** too.

## Installation
```
pip install git+https://github.com/wejoncy/GPTQ-for-LLMs.git@aaf4e73eff732f85974cf7f03245901007c039dc
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
