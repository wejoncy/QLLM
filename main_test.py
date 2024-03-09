import sys
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
sys.path.insert(0, "/home/jicwen/work/onnxruntime/build/Linux/Debug/build/lib/")
import onnxruntime
from qllm import run as qllm_cli

# autograd_inlining=False

# from transformers import AutoTokenizer

# tokenizer = AutoTokenizer.from_pretrained("mix_1a1", use_fast=True, trust_remote_code=True)
# sample_inputs = torch.load("sample_inputs.pt")
# sample_inputs = tokenizer("hey", return_tensors="pt")
# inputs = {'input_ids': sample_inputs.input_ids.cpu().numpy(), 'attention_mask': sample_inputs.attention_mask.numpy()}

# session = onnxruntime.InferenceSession("mix1a/model.onnx", providers=['CUDAExecutionProvider'])
# o = session.run(None,inputs)
# sys.argv = ["", "--load=../Mixtral-8x7B-v0.1-GPTQ/", "--nsamples=4", "--use_plugin"]#, '--export_onnx=./mix_1a']
sys.argv = ["", "--model=../Mixtral-8x7B-v0.1/", "--wbits=16", "--nsamples=4", "--use_plugin", ]  # , '--export_onnx=./mix_1a']
# sys.argv = ["", "--model=../Mixtral-8x7B-v0.1/", "--method=hqq", "--wbits=3", "--nsamples=4", "--save=./vllm-hqq", "--use_plugin"]
# sys.argv = ['', '--load=../Carl-33B-GPTQ/', '--wbits=16', '--eval']
sys.argv = ["", "--load=TheBloke/Llama-2-7B-Chat-GPTQ", "--pack_mode=ORT", "--use_plugin", "--export_onnx=./onnx"]
sys.argv = ["", "--load=../Mistral-7B-v0.1-GPTQ", "--pack_mode=ORT" ,"--export_onnx=./onnx", "--use_plugin"]
sys.argv = ['', '--model=../phi-2/', '--wbits=4', '--method=gptq', '--use_plugin',"--pack_mode=ORT", '--export_onnx=./mix1a']  # '--save=hqq4bit',
# sys.argv = ["yesyt", "--model=../Llama-2-7b-chat-hf", '--method=gptq',"--pack_mode=ORT", "--wbits=4", "--dataset=pileval", "--nsamples=16","--export_onnx=./onnx"]
sys.argv = ["", "--load=TheBloke/Llama-2-7B-Chat-AWQ","--export_onnx=./onnx"]
qllm_cli.main()
