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

    def from_pretrained(self, model_name_or_path):
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_name_or_path, use_fast=True, trust_remote_code=True)
        attn_implementation = "flash_attention_2" if os.getenv('USE_FLASH_ATTN', "0")=="1" else None
        return AutoQuantizedModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True,
        attn_implementation=attn_implementation)

    def get_datasets(self, tokenizer, dataset, nsamples, seed):
        return get_sample_datas_for_quantization(tokenizer, dataset, nsamples, seed)

    # you shouldn't rewrite this function
    @torch.no_grad()
    def __dispatch_quant(self, model, inputs_dataloader, config, dev):
        quantizer = quantization.get_quantizer(config)
        quantizer.set_tokenizer(self.tokenizer)
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
        inputs["pad_token_id"] = self.tokenizer.eos_token_id
        out = model.generate(**inputs, max_length=50)

        # from .plugin import perplexity_utils
        # ppl = perplexity_utils.Perplexity(
        #         model,
        #         self.tokenizer,
        #         "wikitext",
        #         None,
        #         "test",
        #         "text",
        #     )
        # ppl.calculate_perplexity(512, 512)

        model.to('cpu')
        print(self.tokenizer.decode(out[0]))

    # TODO: perform packing on GPU
    def pack_model(self, model, quantizers, pack_mode):
        if not quantizers:
            logger.warning("No quantized layers found, skip packing, If you are not using VPTQ, please check the log")
            return model
        attention_layers = find_layers(model, self.quant_layers+[ScaledLinear])
        attention_layers = {n: attention_layers[n] for n in quantizers}

        quant_config_by_layer = {key: {
            "wbits": value[-2], "groupsize": value[-1]} for key, value in quantizers.items()}
        meta_info = model.quant_config.to_meta
        wbits, quant_method = meta_info.bits, meta_info.quant_method
        quant_config_by_layer["quant_method"] = model.quant_config.quant_method

        target_layer = select_quant_linear(pack_mode, wbits, quant_method)

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

        model.quant_config.version = qlayers[name].pack_mode if qlayers else ""
        if os.getenv('COMPATIBLE_WITH_AUTOGPTQ', None) == "1" and pack_mode == "GPTQ":
            model.quant_config["COMPATIBLE_WITH_AUTOGPTQ"] = 1
        model.quant_config_by_layer = quant_config_by_layer

        return model

    def repack_to_new_mode(self, model, new_pack_mode):
        old_pack_mode = model.quant_config.version
        if old_pack_mode == new_pack_mode:
            return model
        meta_info = model.quant_config.to_meta
        bits = meta_info.bits
        source_layer = select_quant_linear(old_pack_mode, bits, meta_info.quant_method)
        target_layer = select_quant_linear(new_pack_mode, bits, meta_info.quant_method)
        if source_layer == target_layer:
            return model
        model.quant_config.version = new_pack_mode
        qlayers = find_layers(model, [source_layer])
        for module_name, qlayer in tqdm.tqdm(qlayers.items(),
                desc=f"repacking model from pack_mode=`{old_pack_mode}` to `{new_pack_mode}`"):
            fp16_weight, scales, zeros = qlayer.unpack()
            qlayer.weight = fp16_weight
            new_module = target_layer(
                qlayer.bits,
                qlayer.groupsize,
                qlayer.infeatures,
                qlayer.outfeatures,
                qlayer.bias is not None,
                dtype=qlayer.weight.dtype,
        )
            new_module.bias = qlayer.bias if qlayer.bias is not None else None
            set_op_by_name(model, module_name, new_module)
            new_module.pack(qlayer, scales.T, zeros.T, qlayer.g_idx)
            del qlayer.weight
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
            model = self.from_pretrained(model_or_model_path)
        else:
            model = model_or_model_path
            self.tokenizer = tokenizer
            assert tokenizer is not None, "tokenizer is required when model is provided"
        from .args_config import FakeArgs
        args = FakeArgs(**kwargs)
        config = quantization.build_config(args)

        inputs_dataloader = self.get_datasets(self.tokenizer, args.dataset, args.nsamples, args.seed)
        quantizers = self.__dispatch_quant(model, inputs_dataloader, config, dev=torch.device("cuda:0"))
        model = self.pack_model(model, quantizers, args.pack_mode)
        return model

    def run(self, args):
        from .utils.comm_utils import set_seed
        set_seed(args.seed)

        if args.quant_method == "vptq" and (args.quant_config.name == "help" or
                                            not args.quant_config.exists()) :
            example_config = quantization.config_builder.VPTQConfig().to_dict()
            example_config['model_name'] = args.model
            logger.info("An example VPTQ config looks like:\n" + json.dumps(example_config, indent=4))
            return

        if args.pack_mode == "AUTO" and args.allow_mix_bits:
            assert args.quant_method == "gptq", "only gptq support allow_mix_bits mode"
            args.pack_mode = "GPTQ"
        if args.allow_mix_bits and args.pack_mode != "GPTQ":
            raise ValueError("allow_mix_bits only support GPTQ packing mode")
        if not isinstance(args.load,  str):
            args.load = args.load.as_posix()

        if args.quant_method == "awq" and args.nsamples > 64:
            logger.warning("as the memory blast, AWQ will limit to 32 samples for quantization")
            args.nsamples = 64

        if args.tokenizer == "":
            args.tokenizer = args.load if args.load else args.model

        if args.load and args.model:
            args.model = ""
            logger.warning(
                f"--model={args.model} will be ignored when --load is specified")
        elif not args.load and not args.model:
            raise ValueError("either --model or --load must be specified. \
Please run with `-h` to refer the usage.")
        model = self.from_pretrained(args.load + args.model)
        model.eval()

        if not args.load and args.wbits < 16:
            if args.quant_method in ["hqq", "vptq"]:
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
            quantizers = self.__dispatch_quant(model, inputs_dataloader, config, torch.device("cuda:0"))
            model = self.pack_model(model, quantizers, args.pack_mode)
            logger.info(f"Finished quantization and packing weight, time cost:{time.time() - tick}")

        if args.save:
            def repack_func(): return self.repack_to_new_mode(model, args.pack_mode)
            AutoQuantizedModelForCausalLM.save_pretrained(model, self.tokenizer, args.save,
                                                          args.pack_mode, repack_func)

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
