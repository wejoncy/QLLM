import os

import torch
import time
import json
import tqdm
import transformers

from .auto_datasets import get_sample_datas_for_quantization
from .utils import find_layers
from .utils.modelutils import ScaledLinear, make_mixbits_quant_linear, select_quant_linear, set_op_by_name
from .utils.logger import get_logger
from .modeling import AutoQuantizedModelForCausalLM
from . import quantization

logger = get_logger()
ROUNDTRIP_CHECK = False


class AutoModelQuantization(object):
    def __init__(self) -> None:
        super().__init__()
        self.quant_layers = [torch.nn.Linear]
        self.tokenizer = None

    def get_torch_model(self, model_name_or_path):
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_name_or_path, use_fast=True, trust_remote_code=True)
        return AutoQuantizedModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True)

    def get_datasets(self, tokenizer, dataset, nsamples, seed):
        return get_sample_datas_for_quantization(tokenizer, dataset, nsamples, seed)

    def __load_quant(self, quant_model_name_or_path):
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(quant_model_name_or_path, use_fast=True, trust_remote_code=True)
        return AutoQuantizedModelForCausalLM.from_quantized(quant_model_name_or_path, trust_remote_code=True)

    # you shouldn't rewrite this function
    @torch.no_grad()
    def __dispatch_quant(self, model, inputs_dataloader, config, dev):
        quantizer = quantization.get_quantizer(config)
        return quantizer.quantize(model, inputs_dataloader, dev)

    @torch.inference_mode()
    def eval_model(self, model, pack_mode, dev):
        logger.info('Evaluating ...')

        from .modeling.q_layers.quant_linear_awq import has_awq_inference_engine
        if pack_mode != "GEMM" or has_awq_inference_engine():
            model = self.repack_to_new_mode(model, pack_mode)
            
        if not has_awq_inference_engine() and model.quant_config.version == "GEMM":
            logger.warning(
                "AWQ inference engine not found, will convert to GPTQ packing for inference.")
            model = self.repack_to_new_mode(model, "GPTQ")

        model.eval()
        model.to(dev)

        inputs = self.tokenizer(
            "compared with awq, gptq is", return_tensors="pt").to(model.device)
        out = model.generate(**inputs, max_length=50)

        model.to('cpu')
        print(self.tokenizer.decode(out[0]))

    # TODO: perform packing on GPU
    def pack_model(self, model, quantizers, pack_mode):
        attention_layers = find_layers(model, self.quant_layers+[ScaledLinear])
        attention_layers = {n: attention_layers[n] for n in quantizers}

        quant_config_by_layer = {key: {
            "wbits": value[-2], "groupsize": value[-1]} for key, value in quantizers.items()}
        meta_info = model.quant_config.to_meta
        wbits, method = meta_info.bits, meta_info.method
        quant_config_by_layer["method"] = model.quant_config.method

        target_layer = select_quant_linear(pack_mode, wbits, method)

        make_mixbits_quant_linear(model, quantizers, quant_config_by_layer, target_layer=target_layer)
        qlayers = find_layers(model, [target_layer])
        for name in tqdm.tqdm(qlayers, desc='Packing weights....'):
            quantizers[name], scale, zero, g_idx, _, _ = quantizers[name]
            # rewrite weight as quantized
            if ROUNDTRIP_CHECK:
                qlayers[name].orig_fp_weight = qlayers[name].weight_qdq(
                    attention_layers[name], scale, zero, g_idx).cuda()
                attention_layers[name].weight.data = qlayers[name].orig_fp_weight
                assert (qlayers[name].orig_fp_weight == qlayers[name].weight_qdq(
                    attention_layers[name], scale, zero, g_idx).cuda()).all()

            qlayers[name].pack(attention_layers[name], scale, zero, g_idx)

        model.quant_config.version = qlayers[name].pack_mode
        if os.getenv('COMPATIBLE_WITH_AUTOGPTQ', None) == "1" and pack_mode == "GPTQ":
            model.quant_config["COMPATIBLE_WITH_AUTOGPTQ"] = 1
        model.quant_config_by_layer = quant_config_by_layer

        return model

    def repack_to_new_mode(self, model, new_pack_mode):
        old_pack_mode = model.quant_config.version
        if old_pack_mode == new_pack_mode:
            return model
        meta_info = model.quant_config.to_meta
        bits, groupsize = meta_info.bits, meta_info.group_size
        source_layer = select_quant_linear(old_pack_mode, bits, meta_info.method)
        target_layer = select_quant_linear(new_pack_mode, bits, meta_info.method)
        if source_layer == target_layer:
            return model
        model.quant_config.version = new_pack_mode
        qlayers = find_layers(model, [source_layer])
        for module_name, qlayer in tqdm.tqdm(qlayers.items(),
                desc=f"repacking model from pack_mode=`{old_pack_mode}` to `{new_pack_mode}`"):
            fp16_weight, scales, zeros = qlayer.unpack()
            qlayer.weight = fp16_weight
            new_module = target_layer(bits, groupsize, qlayer.infeatures, qlayer.outfeatures, qlayer.bias is not None)
            new_module.bias = qlayer.bias if qlayer.bias is not None else None
            set_op_by_name(model, module_name, new_module)
            new_module.pack(qlayer, scales.T, zeros.T, qlayer.g_idx)
            qlayer.to('cpu')
            new_module.to('cpu')
        del qlayers
        torch.cuda.empty_cache()
        return model

    @torch.no_grad()
    def export_onnx(self, model: torch.nn.Module, onnx_path_str: str,
                    sample_inputs: tuple, with_past: bool = False):
        # enforce to ort pack_mode
        pack_mode = model.quant_config.version
        wbits = model.quant_config.to_meta.bits
        if pack_mode != "ORT":
            if wbits != 4:
                logger.warn("ORT pack_mode only support 4bit quantization, will use the original pack mode")
            else:
                model = self.repack_to_new_mode(model, "ORT")

        from .utils.onnx import exporter
        opset = 16
        if self.tokenizer:
            sample_inputs = self.tokenizer("Hello world", return_tensors="pt")
            sample_inputs = (sample_inputs.input_ids, sample_inputs.attention_mask)
            self.tokenizer.save_pretrained(onnx_path_str)
        model.config.to_json_file(f"{onnx_path_str}/config.json")
        model.generation_config.to_json_file(f"{onnx_path_str}/generation_config.json")
        onnx_model_path = exporter.export_onnx(model, onnx_path_str, sample_inputs, with_past, opset)

        # verify correctness
        exporter.verify_correcness(model, sample_inputs, onnx_model_path, with_past)

    def api_quantize(self, model_or_model_path, tokenizer=None, **kwargs):
        if not isinstance(self, AutoModelQuantization):
            raise ValueError("api_quantize should be called by AutoModelQuantization instance")
        if isinstance(model_or_model_path, str):
            model = self.get_torch_model(model_or_model_path)
        else:
            model = model_or_model_path
            self.tokenizer = tokenizer
            assert tokenizer is not None, "tokenizer is required when model is provided"
        from .args_config import FakeArgs
        args = FakeArgs(**kwargs)
        config = quantization.build_config(args)

        inputs_dataloader = self.get_datasets(self.tokenizer, args.dataset, args.nsamples, args.seed)
        quantizers = self.__dispatch_quant(model, inputs_dataloader, config, "cuda")
        model = self.pack_model(model, quantizers, args.pack_mode)
        return model

    def run(self, args):
        from .utils.comm_utils import set_seed
        set_seed(args.seed)

        if args.pack_mode == "AUTO" and args.allow_mix_bits:
            assert args.method == "gptq", "only gptq support allow_mix_bits mode"
            args.pack_mode = "GPTQ"
        if args.allow_mix_bits and args.pack_mode != "GPTQ":
            raise ValueError("allow_mix_bits only support GPTQ packing mode")
        if not isinstance(args.load,  str):
            args.load = args.load.as_posix()

        if args.method == "awq" and args.nsamples > 64:
            logger.warning("as the memory blast, AWQ will limit to 32 samples for quantization")
            args.nsamples = 64

        if args.tokenizer == "":
            args.tokenizer = args.load if args.load else args.model

        if args.load:
            if args.model != "":
                logger.warn(
                    f"--model={args.model} will be ignored when --load is specified")
            model = self.__load_quant(args.load)
            model.eval()
        elif args.model:
            model = self.get_torch_model(args.model)
            model.eval()
        else:
            raise ValueError("either --model or --load must be specified. \
Please run with `-h` to refer the usage.")

        if not args.load and args.wbits < 16:
            if args.method == "hqq":
                inputs_dataloader = None
            else:
                inputs_dataloader = self.get_datasets(args.tokenizer, args.dataset, args.nsamples, args.seed)
            if args.mix_qlayer_conf:
                with open(args.mix_qlayer_conf) as fp:
                    args.mix_qlayer_conf = json.load(fp)
            else:
                args.mix_qlayer_conf = {}
            tick = time.time()
            config = quantization.build_config(args)
            quantizers = self.__dispatch_quant(model, inputs_dataloader, config, "cuda")
            model = self.pack_model(model, quantizers, args.pack_mode)
            logger.info(f"Finished quantization and packing weight, time cost:{time.time() - tick}")

        if args.save:
            def repack_func(): return self.repack_to_new_mode(model, args.pack_mode)
            AutoQuantizedModelForCausalLM.save_pretrained(model, self.tokenizer, args.save,
                                                          args.pack_mode, repack_func, safe_serialization=False)

        if args.eval:
            self.eval_model(model, args.pack_mode, "cuda")

        if args.export_onnx:
            inputs_dataloader = self.get_datasets(
                args.tokenizer, args.dataset, args.nsamples, args.seed) if self.tokenizer is None else [None]
            self.export_onnx(model, args.export_onnx, inputs_dataloader[0], True)

        if args.use_plugin:
            from .plugin.conversation import loop_in_chat_completion
            from .modeling.q_layers.ext_package_checker import is_the_machine_support_awq_engine
            if args.wbits < 16 and not is_the_machine_support_awq_engine(args.wbits
                                                                         ) and model.quant_config.version == "GEMM":
                logger.warning("AWQ inference engine not found, will convert to GPTQ packing for inference.")
                model = self.repack_to_new_mode(model, "GPTQ")
            loop_in_chat_completion(self.tokenizer, model)
