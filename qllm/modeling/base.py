import torch
import transformers
from transformers import AutoModelForCausalLM
from pathlib import Path
import tqdm
import glob
import json
import contextlib
import torch
import accelerate
from typing import Dict, List, Optional, Union
from transformers.utils.hub import cached_file


from .. import utils
from .config import BaseQuantizeConfig
logger = utils.logger.get_logger()


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

class AutoQuantizedModelForCausalLM:
    def __init__(self):
        raise EnvironmentError(
            "AutoQuantizedModelForCausalLM is designed to be instantiated\n"
            "using `AutoQuantizedModelForCausalLM.from_pretrained` if want to quantize a pretrained model.\n"
            "using `AutoQuantizedModelForCausalLM.from_quantized` if want to inference with quantized model."
        )

    @staticmethod
    def disable_double_init():
        # prevent double init of huggingface
        utils.comm_utils.disable_huggingface_init()


    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        max_memory: Optional[dict] = None,
        trust_remote_code: bool = False,
        **model_init_kwargs
    ) -> AutoModelForCausalLM:
        if pretrained_model_name_or_path is None:
            raise ValueError("model_name_or_path must be specified.")
        logger.info(f"loading model from {pretrained_model_name_or_path}")
        cls.disable_double_init()

        llm = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path, torch_dtype=torch.float16, trust_remote_code=trust_remote_code)
        return llm


    @classmethod
    def from_quantized(
        cls, 
        model_name_or_path: Optional[str],
        device_map: Optional[Union[str, Dict[str, Union[int, str]]]] = "balanced",
        max_memory: Optional[dict] = None,
        device: Optional[Union[str, int]] = None,
        low_cpu_mem_usage: bool = False,
        use_triton: bool = False,
        torch_dtype: Optional[torch.dtype] = torch.float16,
        quant_config: Optional[BaseQuantizeConfig] = None,
        use_safetensors: bool = True,
        trust_remote_code: bool = False,
        warmup_triton: bool = False,
        **kwargs) -> AutoModelForCausalLM:

        args = kwargs.pop("args", None)
        cls.disable_double_init()
        
        if isinstance(device_map, str):
            assert device_map in ["auto", "balanced", "balanced_low_0", "sequential"], \
                  'device_map must be auto, balanced, balanced_low_0 or sequential'

        if model_name_or_path is None:
            raise ValueError("model_name_or_path must be specified.")
        logger.info(f"loading quantized model from {model_name_or_path}")
        init_contexts = [transformers.modeling_utils.no_init_weights(),
                         #accelerate.init_empty_weights(include_buffers=False)
        ]
        with transformers.utils.generic.ContextManagers(init_contexts):
            model = AutoModelForCausalLM.from_config(
                transformers.AutoConfig.from_pretrained(
                    model_name_or_path,trust_remote_code=trust_remote_code, dtype=torch_dtype),
                trust_remote_code=trust_remote_code)

        if quant_config is None:
            quant_config = BaseQuantizeConfig.from_pretrained(model_name_or_path, args)
        model.quant_config = quant_config.quant_config
        model.quant_config_by_layer = quant_config.quant_config_by_op

        quant_layers = [torch.nn.Linear]
        layers = utils.find_layers(model, layers=quant_layers)

        # all layers has the same quantization config
        if 'groupsize' not in quant_config.quant_config_by_op:
            for layer_name in list(layers.keys()):
                if layer_name not in quant_config.quant_config_by_op:
                    del layers[layer_name]
        else: # removed unquantized layer, TODO load layers from safetensors
            for layer_name in list(layers.keys()):
                if len(layer_name.split('.')) <= 3:
                    del layers[layer_name]

        target_layer = utils.modelutils.select_quant_linear(
            quant_config.quant_config["version"], quant_config.wbits())
        utils.modelutils.make_mixbits_quant_linear(
            model, layers, quant_config.quant_config_by_op, target_layer=target_layer)
        if quant_config.method == "awq":
            from ..quantization.quant_awq import scale_activations
            scale_activations(model)
        del layers
        
        model.tie_weights()
        try:
            accelerate.big_modeling.load_checkpoint_and_dispatch(
                model,
                checkpoint=model_name_or_path,
                device_map=device_map,
                no_split_module_classes=None,
                offload_folder=None,
                dtype=torch_dtype,
            )
        except Exception as e:
            import safetensors
            if Path(model_name_or_path).exists():  # local
                while True:
                    weight_bins = glob.glob(
                        str(Path(model_name_or_path).absolute()/'pytorch_model*.bin'))
                    if len(weight_bins) > 0:
                        for i in tqdm.tqdm(range(len(weight_bins)), desc="loading weights"):
                            model.load_state_dict(torch.load(weight_bins[i]), strict=False)
                        break
                    weight_bins = glob.glob(str(Path(model_name_or_path).absolute()/'*.safetensors'))
                    if len(weight_bins) > 0:                        
                        for i in tqdm.tqdm(range(len(weight_bins)), desc="loading weights"):
                            model.load_state_dict(safetensors.torch.load_file(
                                weight_bins[i], device="cpu"), strict=False)
                        break
                    raise ValueError(f"{model_name_or_path} is not a folder containing weights or safetensors")
            else:
                index_config_file = quant_config.get_resolved_base_dir(model_name_or_path, "model.safetensors.index.json")
                if index_config_file:
                    index_files = set(json.load(open(index_config_file))["weight_map"].values())
                    for index_file in tqdm.tqdm(index_files, desc="loading weights"):
                        weight_file = cached_file(model_name_or_path, index_file)
                        model.load_state_dict(safetensors.torch.load_file(
                            weight_file, device="cpu"), strict=False)
                else:
                    weight_file = cached_file(model_name_or_path, "model.safetensors")
                    model.load_state_dict(safetensors.torch.load_file(weight_file, device="cpu"), strict=False)
                    
            # weight_dict = torch.load(weight_bins[0])
            # for i in range(1, len(weight_bins)):
            #    weight_dict.update(torch.load(weight_bins[i]))
            # model.load_state_dict(weight_dict)
            # layers.autotune_warmup_linear(model, transpose=False)

        # autogptq has extra -1 in qzeros but we don't have it.
        if quant_config.compatible_with_autogptq:
            qlayers = utils.find_layers(model, [target_layer])
            for module_name, qlayer in tqdm.tqdm(qlayers.items(), desc="Repacking AutoGPTQ qzeros..."):
                qlayer.handle_qzeros_for_autogptq()
            import os
            os.environ["compatible_with_autogptq"] = '0'
            # The kernel with compatible_with_autogptq has some bugs now in XbitOps, let's always replace qzeros

        return model

    @staticmethod
    def save_pretrained(model, tokenizer, save_directory: Union[str, Path], pack_mode:str, repack_func, save_serialization: bool = False):
        quant_config_by_layer, quant_config = model.quant_config_by_layer,model.quant_config
        if pack_mode != quant_config["version"] and pack_mode != "AUTO":
            repack_func()

        model.save_pretrained(save_directory, save_serialization=save_serialization)
        tokenizer is not None and tokenizer.save_pretrained(save_directory)

        open(save_directory+"/quant_config_by_layer.json",
                'w').write(json.dumps(quant_config_by_layer))
        open(save_directory+"/quantize_config.json", 'w').write(json.dumps(quant_config))
