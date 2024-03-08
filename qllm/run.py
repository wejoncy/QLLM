import argparse

from .auto_model_quantization import AutoModelQuantization
from .args_config import FakeArgs

def define_basic_args():
    parser = argparse.ArgumentParser(description="""
A general tool to quantize LLMs with the GPTQ/AWQ/HQQ method.
you can easily quantize your model and save to checkpoint, which is compatiable with \
[vLLM](https://github.com/vllm-project/vllm).
You can also test the quantized model with a conversation plugin.

A typical usage is:
    python -m qllm --model  meta-llama/Llama-2-7b-chat-hf  --method=awq  \
--dataset=pileval --nsamples=16  --use_plugin --save ./Llama-2-7b-chat-hf_awq_q4/ \
--export_onnx ./onnx_models/

    method can be 'awq' or 'gptq', 'hqq' """,
                                     formatter_class=argparse.RawTextHelpFormatter)
    default_args = FakeArgs()
    parser.add_argument('--method', type=str, default=default_args.method,
                        choices=["gptq", "awq", "hqq"], help='the quantization method')
    parser.add_argument('--model', type=str, default="",
                        help='float/float16 model to load, such as [mosaicml/mpt-7b]')
    parser.add_argument('--tokenizer', type=str, default="", help='default same as [model]')
    parser.add_argument('--dataset', type=str, default=default_args.dataset,
                        choices=['wikitext2', 'ptb', 'c4', 'ptb-new', 'c4-new', "pileval"], help='Where to extract calibration data from.')
    parser.add_argument('--seed', type=int, default=default_args.seed, help='Seed for sampling the calibration data.')
    parser.add_argument('--nsamples', type=int, default=default_args.nsamples, help='Number of calibration data samples.')
    parser.add_argument('--percdamp', type=float, default=default_args.percdamp,
                        help='Percent of the average Hessian diagonal to use for dampening.')
    parser.add_argument(
            '--static-groups', action='store_true',
            help='(gptq only.) Whether to use static groups; recommended when using `--actorder` for more efficient inference.'
        )
    parser.add_argument('--wbits', type=int, default=default_args.wbits,
                        choices=[2, 3, 4, 5, 6, 7, 8, 16], help='#bits to use for quantization; use 16 for evaluating base model.')
    parser.add_argument('--mix_qlayer_conf', type=str, default=None,
                        help='Mix quantization layer configuration.(groupsize,wbits)')
    parser.add_argument('--groupsize', type=int, default=default_args.groupsize,
                        help='Groupsize to use for quantization; -1 uses full row.')
    parser.add_argument('--eval', action='store_true', help='evaluate quantized model.')
    parser.add_argument('--save', type=str, default='', help='Save quantized checkpoint under this name.')
    parser.add_argument('--save_safetensors', type=str, default='',
                        help='Save quantized `.safetensors` checkpoint under this name.')
    parser.add_argument('--load', type=str, default='', help='Load quantized model.')
    parser.add_argument('--sym', action='store_true', help='Whether to perform symmetric quantization.')
    parser.add_argument('--act-order', action='store_true',
                        help='Whether to apply the activation order GPTQ heuristic')
    parser.add_argument('--true-sequential', action='store_true', help='Whether to run in true sequential model.')
    parser.add_argument('--allow_mix_bits',action='store_true',
                        help='Auto upgrade layer precision to higher precision, for example int2 to int4, groupsize 128 to 64.')
    parser.add_argument('--export_onnx', type=str, default=None, help='where does the onnx model save to.')
    parser.add_argument('--use_plugin', action='store_true', help='test with plugin, such as fastchat conversation')
    parser.add_argument('--pack_mode', type=str, default=default_args.pack_mode,
                        choices=["AUTO", "GEMM", "GPTQ", "ORT", "HQQ"], help="""the quantization pack mode, 
`GEMM` represents to use AWQ GEMM engine, same as what AutoAWQ used, 
`AUTO` represents that it is selected by the GPU arch, as awq GEMM needs SM75+ 
`GPTQ` represent using old GPTQ style DQ+GEMM, it is not fused but more general than AWQ-GEMM, 
and can be used on all GPU archs.
`ORT` represents using onnxruntime packing stype,
""")

    return parser


def main():
    parser = define_basic_args()
    args = parser.parse_args()
    print(args)

    model_quanter = AutoModelQuantization()
    model_quanter.run(args)


if __name__ == '__main__':
    main()
