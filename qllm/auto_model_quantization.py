import os
# if "CUDA_VISIBLE_DEVICES" not in os.environ: # NOQA
#    os.environ["CUDA_VISIBLE_DEVICES"] = "1" # NOQA

import torch.nn as nn
import torch
import numpy as np
import time
from pathlib import Path
import json
import tqdm

from .auto_datasets import get_sample_datas_for_quantization
from .utils import find_layers, DEV
from .utils.modelutils import ScaledLinear, make_mixbits_quant_linear, select_quant_linear, set_op_by_name
from .utils.logger import get_logger
from .modeling import AutoQuantizedModelForCausalLM

logger = get_logger()
ROUNDTRIP_CHECK = False

class AutoModelQuantization(object):
    def __init__(self) -> None:
        super().__init__()
        self.quant_layers = [torch.nn.Linear]
        self.tokenizer = None

    def get_torch_model(self, args, dev='cpu'):
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.model, use_fast=True, trust_remote_code=True)
        return AutoQuantizedModelForCausalLM.from_pretrained(args.model, args=args, trust_remote_code=True)

    def get_datasets(self, args):
        return get_sample_datas_for_quantization(args)


    def __load_quant(self, args):
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(args.load, use_fast=True, trust_remote_code=True)
        return AutoQuantizedModelForCausalLM.from_quantized(args.load, args=args, trust_remote_code=True)

    # you shouldn't rewrite this function
    @torch.no_grad()
    def __quant_by_sequential(self, model, inputs_dataloader, args, dev):
        from .quantization import get_quantizer
        quantizer = get_quantizer(args)
        return quantizer.quantize(model, inputs_dataloader, dev)

    @torch.inference_mode()
    def eval_model(self, model, dev, args):
        logger.info('Evaluating ...')

        from .modeling.q_layers.quant_linear_awq import has_awq_inference_engine
        if not has_awq_inference_engine() and model.quant_config["version"] == "GEMM":
            logger.warning("AWQ inference engine not found, will convert to GPTQ packing for inference.")
            model = self.repack_to_new_mode(model, args, "GPTQ")

        model.eval()
        model.to(dev)

        inputs = self.tokenizer("compared with awq, gptq is", return_tensors="pt").to(model.device)
        out = model.generate(**inputs, max_length=50)
        
        model.to('cpu')
        print(self.tokenizer.decode(out[0]))

    # TODO: perform packing on GPU
    def pack_model(self, model, quantizers, args):
        attention_layers = find_layers(model, self.quant_layers+[ScaledLinear])
        attention_layers = {n: attention_layers[n] for n in quantizers}

        quant_config_by_layer = {key: {"wbits": value[-2], "groupsize": value[-1]} for key, value in quantizers.items()}
        quant_config_by_layer["method"] = args.method
        
        target_layer = select_quant_linear(args.pack_mode, args.wbits)

        make_mixbits_quant_linear(
            model, quantizers, quant_config_by_layer, target_layer=target_layer)
        qlayers = find_layers(model, [target_layer])
        for name in tqdm.tqdm(qlayers, desc='Packing weights....'):
            quantizers[name], scale, zero, g_idx, _, _ = quantizers[name]
            # rewrite weight as quantized
            if ROUNDTRIP_CHECK:
                qlayers[name].orig_fp_weight = qlayers[name].weight_qdq(attention_layers[name], scale, zero, g_idx).cuda()
                attention_layers[name].weight.data = qlayers[name].orig_fp_weight
                assert (qlayers[name].orig_fp_weight == qlayers[name].weight_qdq(
                    attention_layers[name], scale, zero, g_idx).cuda()).all()

            qlayers[name].pack(attention_layers[name], scale, zero, g_idx)

        model.quant_config["version"] = qlayers[name].pack_mode
        model.quant_config_by_layer = quant_config_by_layer

        return model

    def repack_to_new_mode(self, model, args, new_pack_mode):
        old_pack_mode = model.quant_config["version"]
        model.quant_config["version"] = new_pack_mode
        bits, groupsize = args.wbits, args.groupsize
        source_layer = select_quant_linear(old_pack_mode, args.wbits)
        target_layer = select_quant_linear(new_pack_mode, args.wbits)
        qlayers = find_layers(model, [source_layer])
        for module_name, qlayer in tqdm.tqdm(qlayers.items(), desc=f"replacing model packed-weight from pack_mode=`{old_pack_mode}` to `{new_pack_mode}`"):
            fp16_weight, scales, zeros = qlayer.unpack()
            qlayer.weight = fp16_weight
            tmp = qlayer
            new_module = target_layer(bits, groupsize, tmp.infeatures, tmp.outfeatures, tmp.bias is not None)
            set_op_by_name(model, module_name, new_module)
            new_module.pack(tmp, scales.T, zeros.T, None)
            qlayer.to('cpu')
            new_module.to('cpu')
        del qlayers
        torch.cuda.empty_cache()
        return model

    @torch.no_grad()
    def export_onnx(self, model: torch.nn.Module, onnx_path_str: str, sample_inputs: tuple, with_past: bool = False, args=None):
        if args.pack_mode != "ORT" and os.getenv("KEEP_GPTQ_PACK") != "1":
            model = self.repack_to_new_mode(model, args, "ORT")
        from .utils.onnx import exporter
        opset = 16
        onnx_model_path = exporter.export_onnx(model, onnx_path_str, sample_inputs, with_past, opset)
        self.tokenizer is not None and self.tokenizer.save_pretrained(onnx_path_str)

        #verify correctness
        import onnxruntime
        session = onnxruntime.InferenceSession(onnx_model_path, providers=['CUDAExecutionProvider'])
        mask = np.ones(sample_inputs[0].shape, dtype=np.int64) if sample_inputs[1] is None else sample_inputs[1].cpu().numpy()
        num_layers = model.config.num_hidden_layers
        inputs = {'input_ids': sample_inputs[0].cpu().numpy(), 'attention_mask': mask, 'use_cache_branch': np.array([0], dtype=np.bool_)}
        for i in range(num_layers):
            inputs[f'present_key.{i}'] = np.zeros((1, 32, 32, 128), dtype=np.float16)
            inputs[f'present_values.{i}'] = np.zeros((1, 32, 32, 128), dtype=np.float16)
        outputs = session.run(None, inputs)
        ref = model(sample_inputs[0].cuda())
        err = ref.logits.cpu().numpy()-outputs[0]
        print("max abs err:", np.abs(err).max(), "correctness check ",
              "" if np.abs(err).max() < 1e-2 else "not", " passed")


    def run(self, args):
        if args.pack_mode == "AUTO" and args.allow_mix_bits:
            assert args.method == "gptq", "only gptq support allow_mix_bits mode"
            args.pack_mode = "GPTQ"
        if args.allow_mix_bits and args.pack_mode != "GPTQ":
            raise ValueError("allow_mix_bits only support GPTQ packing mode")
        if type(args.load) is not str:
            args.load = args.load.as_posix()

        if args.tokenizer == "":
            args.tokenizer = args.load if args.load else args.model

        if args.load:
            if args.model != "":
                logger.warn(f"--model={args.model} will be ignored when --load is specified")
            model = self.__load_quant(args)
            model.eval()
        elif args.model:
            model = self.get_torch_model(args, dev='cpu')
            model.eval()
        else:
            raise ValueError("either --model or --load must be specified. \
Please run with `-h` to refer the usage.")

        if args.export_onnx or (not args.load and args.wbits < 16):
            inputs_dataloader = self.get_datasets(args)

        if not args.load and args.wbits < 16:
            if args.mix_qlayer_conf:
                args.mix_qlayer_conf = json.load(open(args.mix_qlayer_conf))
            else:
                args.mix_qlayer_conf = {}
            tick = time.time()
            quantizers = self.__quant_by_sequential(model, inputs_dataloader, args, DEV)
            model = self.pack_model(
                model, quantizers, args)
            logger.info(f"Finished quantization and packing weight, time cost:{time.time() - tick}")

        if args.save:
            repack_func = lambda: self.repack_to_new_mode(model, args, args.pack_mode)
            AutoQuantizedModelForCausalLM.save_pretrained(model, self.tokenizer, args.save, 
                                                          args.pack_mode, repack_func, save_serialization=False)

        if args.eval:
            self.eval_model(model, DEV, args)

        if args.export_onnx:
            self.export_onnx(model, args.export_onnx, inputs_dataloader[0], True, args=args)

        if args.use_plugin:
            from .plugin.conversation import loop_in_chat_completion
            from .modeling.q_layers.quant_linear_awq import has_awq_inference_engine
            if not has_awq_inference_engine() and model.quant_config["version"] == "GEMM":
                logger.warning("AWQ inference engine not found, will convert to GPTQ packing for inference.")
                model = self.repack_to_new_mode(model, args, "GPTQ")
            loop_in_chat_completion(self.tokenizer, model)
