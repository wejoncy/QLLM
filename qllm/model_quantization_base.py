from texttable import Texttable
import os
# if "CUDA_VISIBLE_DEVICES" not in os.environ: # NOQA
#    os.environ["CUDA_VISIBLE_DEVICES"] = "1" # NOQA

import torch.nn as nn
import torch
import numpy as np
import glob
import argparse
import time
from pathlib import Path
import json
import sys
import contextlib
import tqdm

from . import utils
from .utils import find_layers, DEV, export_quant_table
from .utils.modelutils import ScaledLinear, make_mixbits_quant_linear, select_quant_linear, set_op_by_name
from .utils.logger import get_logger

logger = get_logger()
NEED_CHECK_PACK = False


def remove_unquantized_layers(qmodel, layers, args):
    # backward compatability
    if not (Path(qmodel)/"quant.op.json").exists():
        quant_layers_json = {layer_name: {"groupsize": args.groupsize, "wbits": args.wbits}
                                for layer_name in layers.keys() if len(layer_name.split('.')) > 3}
        quant_layers_json["method"] = args.method
        open(Path(qmodel)/"quant.op.json", "w").write(json.dumps(quant_layers_json))

    # load quant info
    with open(Path(qmodel)/"quant.op.json") as fp:
        qunat_info = json.load(fp)
        args.method = qunat_info["method"]
        args.qunat_info = qunat_info
    for layer_name in list(layers.keys()):
        if layer_name not in qunat_info:
            del layers[layer_name]

def load_quant_config(qmodel, args):
    if (Path(qmodel)/"quant_config.json").exists():
        quant_config = json.load(open(Path(qmodel)/"quant_config.json"))
        args.wbits = quant_config["w_bit"]
        args.groupsize = quant_config["q_group_size"]
    elif (Path(qmodel)/"quantize_config.json").exists(): #GPTQ-for-llama/AutoGPTQ
        quant_config = json.load(open(Path(qmodel)/"quantize_config.json"))
        args.wbits = quant_config["bits"]
        args.groupsize = quant_config["group_size"]
    else:
        raise ValueError("quant_config.json not found in checkpoint directory")
    pack_mode = quant_config["version"]

    if args.pack_mode != quant_config["version"]:
        logger.warn(f"pack_mode {args.pack_mode} is not compatiable with checkpoint version" +
                    f"{pack_mode}, will force to use the checkpoint version {pack_mode}")
        args.pack_mode = pack_mode

class ModelQuantizationBase(object):
    def __init__(self) -> None:
        super().__init__()
        self.quant_layers = [torch.nn.Linear]

    def get_torch_model(self, args, dev='cpu'):
        logger.info(f"loading model from {args.model}")
        utils.comm_utils.disable_huggingface_init()
        if args.load:
            import transformers
            llm = transformers.AutoModelForCausalLM.from_config(transformers.AutoConfig.from_pretrained(args.model)).half()
        else:
            from transformers import AutoModelForCausalLM
            llm = AutoModelForCausalLM.from_pretrained(
                args.model, torch_dtype=torch.float16, trust_remote_code=True).to(dev)

        from pathlib import Path
        cache_dir = Path(f"/tmp/qllm_v1/{args.model.replace(' ','_')}_{args.dataset}_dataloader.pt")
        cache_dir.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"loading dataset from {args.dataset}")
        if cache_dir.exists():
            logger.info(f"found cached dataloader in {cache_dir}")
            dataloader = torch.load(cache_dir)
        else:
            dataloader, _ = utils.get_loaders(args.dataset, nsamples=args.nsamples,
                                              seed=args.seed, model=args.tokenizer, seqlen=2048)
            torch.save(dataloader, str(cache_dir))
        return llm, dataloader  # model, dataloader

    def __load_quant(self, qmodel, args):
        logger.info(f"loading quantized model from {qmodel}")
        args.model = args.load

        @contextlib.contextmanager
        def stack_attr(attrs: list):
            old_attr = []
            new_method = lambda x, *args, **kwargs: x
            for attr in attrs:
                try:
                    old_attr.append(getattr(torch.Tensor, attr))
                    setattr(torch.Tensor, attr, new_method)
                except:
                    old_attr.append(None)
            yield
            for idx, attr in enumerate(attrs):
                if old_attr[idx] is not None:
                    setattr(torch.Tensor, attr, old_attr[idx])

        with stack_attr(['normal_', 'uniform_', 'kaiming_uniform_', 'kaiming_normal_']):
            model, dataloader = self.get_torch_model(args, dev='cpu')

        layers = find_layers(model, layers=self.quant_layers)
        remove_unquantized_layers(qmodel, layers, args)
        load_quant_config(qmodel, args)
        
        target_layer = select_quant_linear(args.pack_mode, args.wbits)
        make_mixbits_quant_linear(
            model, layers, args.qunat_info, target_layer=target_layer)
        if args.method == "awq":
            from .quantization.quant_awq import scale_activations
            scale_activations(model)
        del layers
        
        model.tie_weights()
        try:
            import accelerate
            accelerate.load_checkpoint_in_model(
                model,
                checkpoint=qmodel,
                device_map=None,
                offload_folder=None,
                dtype=None
            )
        except:
            while True:
                weight_bins = glob.glob(os.path.abspath(qmodel)+'/pytorch_model*.bin')
                if len(weight_bins) > 0:
                    for i in tqdm.tqdm(range(len(weight_bins)), desc="loading weights"):
                        model.load_state_dict(torch.load(weight_bins[i]), strict=False)
                    break
                weight_bins = glob.glob(os.path.abspath(qmodel)+'/*.safetensors')
                if len(weight_bins) > 0:
                    import safetensors
                    for i in tqdm.tqdm(range(len(weight_bins)), desc="loading weights"):
                        model.load_state_dict(safetensors.torch.load_file(
                            weight_bins[i], device="cpu"), strict=False)
                    break
                raise ValueError(f"{qmodel} is not a folder containing weights or safetensors")
            # weight_dict = torch.load(weight_bins[0])
            # for i in range(1, len(weight_bins)):
            #    weight_dict.update(torch.load(weight_bins[i]))
            # model.load_state_dict(weight_dict)
            # quant.autotune_warmup_linear(model, transpose=False)
        return model, dataloader

    # you shouldn't rewrite this function
    @torch.no_grad()
    def __quant_by_sequential(self, model, dataloader, args, dev):
        from .quantization import get_quantizer
        quantizer = get_quantizer(args)
        return quantizer.quantize(model, dataloader, dev)

    @torch.no_grad()
    def eval_model(self, model, dev):
        logger.info('Evaluating ...')
        logger.warn("you should rewrite this function for your model")

    # TODO: perform packing on GPU
    def pack_model(self, model, quantizers, args):
        attention_layers = find_layers(model, self.quant_layers+[ScaledLinear])
        attention_layers = {n: attention_layers[n] for n in quantizers}
        quant_config = {"zero_point": True, "q_group_size": args.groupsize,
                        "w_bit": args.wbits, "version": args.pack_mode}
        quant_info = {key: {"wbits": value[-2], "groupsize": value[-1]} for key, value in quantizers.items()}
        quant_info["method"] = args.method

        target_layer = select_quant_linear(
            args.pack_mode, args.wbits)

        make_mixbits_quant_linear(model, quantizers, quant_info, target_layer=target_layer)
        qlayers = find_layers(model, [target_layer])
        logger.info('Packing ...')
        for name in qlayers:
            logger.info(name)
            quantizers[name], scale, zero, g_idx, _, _ = quantizers[name]
            # rewrite weight as quantized
            if NEED_CHECK_PACK:
                qlayers[name].oweight = qlayers[name].weight_qdq(attention_layers[name], scale, zero, g_idx).cuda()
                attention_layers[name].weight.data = qlayers[name].oweight
                assert (qlayers[name].oweight == qlayers[name].weight_qdq(
                    attention_layers[name], scale, zero, g_idx).cuda()).all()
                attention_layers[name].nbits = qlayers[name].bits

            qlayers[name].pack_gpu(attention_layers[name], scale, zero, g_idx)

        # quant.make_linear_qdq_back(model,attention_layers)
        # quant.autotune_warmup_linear(model, transpose=False)

        logger.info('Done.')
        return model, quant_info, quant_config

    def re_pack_to_new_mode(self, model, args):
        bits, groupsize = args.wbits, args.groupsize
        source_layer, _ = select_quant_linear(args.pack_mode, args.wbits)
        target_layer, _ = select_quant_linear("ort", args.wbits)
        qlayers = find_layers(model, [source_layer])
        for module_name, qlayer in tqdm.tqdm(qlayers.items(), desc=f"replacing {args.pack_mode} with ort pack_mode"):
            fp16_weight, scales, zeros = qlayer.unpack()
            qlayer.weight = fp16_weight
            tmp = qlayer
            new_module = target_layer(bits, groupsize, tmp.infeatures, tmp.outfeatures, tmp.bias is not None)
            set_op_by_name(model, module_name, new_module)
            new_module.pack_gpu(tmp, scales.T, zeros.T, None)
            qlayer.to('cpu')
        del qlayers
        torch.cuda.empty_cache()
        return model

    @torch.no_grad()
    def export_onnx(self, model: torch.nn.Module, onnx_path_str: str, sample_inputs: tuple, with_past: bool = False, args=None):
        if args.pack_mode != "ORT":
            model = self.re_pack_to_new_mode(model, args)
        from .utils.onnx import exporter
        opset = 16
        exporter.export_onnx(model, onnx_path_str, sample_inputs, with_past, opset)

        #verify correctness
        import onnxruntime
        onnx_model_path = onnx_path_str+'/model_one_for_all.onnx'
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
        print("max abs err:", np.abs(err).max())


    def run(self, args):
        if args.layers_dist:
            gpu_dist = [int(x) for x in args.layers_dist.split(':')]
        else:
            gpu_dist = []

        if type(args.load) is not str:
            args.load = args.load.as_posix()

        if args.tokenizer == "":
            args.tokenizer = args.model if args.model else args.load

        if args.load:
            model, dataloader = self.__load_quant(args.load, args)
            model.eval()
        elif args.model:
            model, dataloader = self.get_torch_model(args, dev='cpu')
            model.eval()
        else:
            raise ValueError("either --model or --load must be specified. \
                Please refer to the usage and run again with correct args.")

        if not args.load and args.wbits < 16 and not args.nearest:
            if args.mix_qlayer_conf:
                args.mix_qlayer_conf = json.load(open(args.mix_qlayer_conf))
            else:
                args.mix_qlayer_conf = {}
            tick = time.time()
            quantizers = self.__quant_by_sequential(model, dataloader, args, DEV)
            model, quant_info, quant_config = self.pack_model(model, quantizers, args)
            logger.info(f"Finished quantization and packing weight, time cost:{time.time() - tick}")

        if args.quant_directory is not None:
            export_quant_table(quantizers, args.quant_directory)

        if not args.observe and args.save:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
            model.save_pretrained(args.save)
            tokenizer.save_pretrained(args.save)

            open(args.save+"/quant.op.json", 'w').write(json.dumps(quant_info))
            open(args.save+"/quant_config.json", 'w').write(json.dumps(quant_config))

        if args.eval:
            self.eval_model(model, DEV)

        if args.export_onnx:
            self.export_onnx(model, args.export_onnx, dataloader[0], True, args=args)

        if args.use_plugin:
            from .plugin.conversation import loop_in_chat_completion
            loop_in_chat_completion(args.tokenizer, model)

        if not args.observe and args.save_safetensors:
            from safetensors.torch import save_file as safe_save
            state_dict = model.state_dict()
            state_dict = {k: v.clone().contiguous() for k, v in state_dict.items()}
            safe_save(state_dict, args.save_safetensors)


