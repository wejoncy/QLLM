import concurrent.futures
import torch
import transformers
from transformers import AutoModelForCausalLM
from pathlib import Path
import tqdm
import glob
import json
import contextlib
import accelerate
from typing import Dict, Optional, Union
from transformers.utils.hub import cached_file
import safetensors
from packaging import version

from .. import utils
from .config import BaseQuantizeConfig
from ..utils.comm_utils import clear_memory

logger = utils.logger.get_logger()


@contextlib.contextmanager
def replace_default_dtype(dtype):
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(old_dtype)


@contextlib.contextmanager
def no_init_weights(attrs: list = None):
    attrs = attrs if attrs is not None else ['normal_', 'uniform_', 'kaiming_uniform_', 'kaiming_normal_']
    old_attr = []
    new_method = lambda x, *args, **kwargs: x
    for attr in attrs:
        try:
            old_attr.append(getattr(torch.Tensor, attr))
            setattr(torch.Tensor, attr, new_method)
        except:  # noqa: E722
            old_attr.append(None)
    yield
    for idx, attr in enumerate(attrs):
        if old_attr[idx] is not None:
            setattr(torch.Tensor, attr, old_attr[idx])

def get_no_split_layer_type_name(model:torch.nn.Module):
   for name,mod in model.named_modules():
       if '.0' in name and name.count('.0') == 1:
           return [mod.__class__.__name__]

def _hf_weight_generator(hf_weights_files, is_safetensors:bool):
    if is_safetensors:
        from safetensors.torch import safe_open
        for st_file in hf_weights_files:
            with safe_open(st_file, framework="pt", device="cuda") as f:
                for name in f.keys():  # noqa: SIM118
                    param = f.get_tensor(name)
                    yield name, param
    else:
        for bin_file in hf_weights_files:
            state = torch.load(bin_file, map_location="cuda")
            for name, param in state.items():
                yield name, param
            del state
            torch.cuda.empty_cache()


def _get_resolved_weight_or_index_file(model_name_or_path):
    if Path(model_name_or_path).exists():  # local
        weight_or_index_file = glob.glob(str(Path(model_name_or_path).absolute()/ '*.index.json'))
        weight_or_index_file += glob.glob(str(Path(model_name_or_path).absolute()/ '*.safetensors'))
        weight_or_index_file += glob.glob(str(Path(model_name_or_path).absolute()/ 'pytorch_model*.bin'))
        if weight_or_index_file: 
            weight_or_index_file = weight_or_index_file[0]
            
        else:
            raise FileNotFoundError("model weight is not found")
    else:
        for possible_index_name in ["model.safetensors.index.json", "pytorch_model.bin.index.json"]:
            weight_or_index_file = BaseQuantizeConfig.get_resolved_base_dir(model_name_or_path, possible_index_name)
            if weight_or_index_file:break
        if not weight_or_index_file:
            for possible_weight_file in ["model.safetensors", "pytorch_model.bin"]:
                weight_or_index_file = cached_file(model_name_or_path, possible_weight_file)
                if weight_or_index_file:break
    return str(weight_or_index_file)


def _load_check_point(model, model_name_or_path, get_keys_only: bool = False):
    weight_or_index_file = _get_resolved_weight_or_index_file(model_name_or_path)
    all_keys = set()
    all_missing_keys = []
    all_unexpected_keys = []
    if weight_or_index_file.endswith(".index.json"):
        with open(weight_or_index_file, "r") as f:
            index = json.loads(f.read())
        if "weight_map" in index:
            index = index["weight_map"]
        checkpoint_files = sorted(list(set(index.values())))
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_checkpoint_files = {executor.submit(cached_file, model_name_or_path, f): f for f in checkpoint_files}
            checkpoint_files = [future.result() for future in concurrent.futures.as_completed(future_to_checkpoint_files)]
        #checkpoint_files = [cached_file(model_name_or_path, f) for f in checkpoint_files]
    else:
        checkpoint_files = [weight_or_index_file]

    #if not get_keys_only:
    #    from ..utils.modelutils import get_op_by_name
    #    params_dict = dict(model.named_parameters())
    #    params_dict.update(dict(model.named_buffers()))
    #    all_missing_keys = list(params_dict.keys())
    #    for name,weight in _hf_weight_generator(checkpoint_files, checkpoint_files[0].endswith("safetensors")):
    #        if name not in params_dict:
    #            all_unexpected_keys.append(name)
    #        else:
    #            #params_dict[name] = weight
    #            op_name, val_name = '.'.join(name.split('.')[:-1]), name.split('.')[-1]
    #            param_dtype = getattr(get_op_by_name(model, op_name), val_name).dtype
    #            new_weight = torch.nn.Parameter(weight.to(param_dtype), requires_grad=False)
    #            setattr(get_op_by_name(model, op_name), val_name, new_weight)
    #            all_missing_keys.remove(name)
    #    return all_missing_keys, all_unexpected_keys

    if len(checkpoint_files) > 0:
        for i in tqdm.tqdm(range(len(checkpoint_files)), desc="loading weights"):
            if not checkpoint_files[i].endswith("safetensors"):
                weights = torch.load(checkpoint_files[i], map_location="cpu")
            else:
                weights = safetensors.torch.load_file(checkpoint_files[i], device="cpu")
            if get_keys_only:
                all_keys.update(weights.keys())
                del weights
            else:
                ret = model.load_state_dict(weights, strict=False)
                del weights
                all_missing_keys.extend(ret.missing_keys)
                all_unexpected_keys.extend(ret.unexpected_keys)
    else:
        raise ValueError(f"{model_name_or_path} is not a folder containing weights or safetensors")

    if get_keys_only:
        return all_keys
    return all_missing_keys, all_unexpected_keys


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
        **kwargs
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
            device_map: Optional[Union[str, Dict[str, Union[int, str]]]] = "auto",
            max_memory: Optional[dict] = None,
            device: Optional[Union[str, int]] = "cuda",
            low_cpu_mem_usage: bool = True,
            use_triton: bool = False,
            torch_dtype: Optional[torch.dtype] = torch.float16,
            use_safetensors: bool = True,
            trust_remote_code: bool = False,
            warmup_triton: bool = False,
            **kwargs) -> AutoModelForCausalLM:

        cls.disable_double_init()

        if isinstance(device_map, str):
            assert device_map in ["auto", "balanced", "balanced_low_0", "sequential"], \
                'device_map must be auto, balanced, balanced_low_0 or sequential'

        if model_name_or_path is None:
            raise ValueError("model_name_or_path must be specified.")
        logger.info(f"loading quantized model from {model_name_or_path}")
        init_contexts = [
            transformers.modeling_utils.no_init_weights(),
            # no_init_weights(),
            replace_default_dtype(torch_dtype),
            # accelerate.init_empty_weights(include_buffers=False)
        ]
        auto_conf = transformers.AutoConfig.from_pretrained(
            model_name_or_path, trust_remote_code=trust_remote_code)
        with transformers.utils.generic.ContextManagers(init_contexts):
            model = AutoModelForCausalLM.from_config(auto_conf, trust_remote_code=trust_remote_code)
        # device_map = accelerate.infer_auto_device_map(
        #    model, dtype=torch_dtype, no_split_module_classes=get_no_split_layer_type_name(model))

        quant_config = BaseQuantizeConfig.from_pretrained(model_name_or_path)
        model.quant_config = quant_config
        model.quant_config_by_layer = quant_config.quant_config_by_op

        quant_layers = [torch.nn.Linear]
        layers = utils.find_layers(model, layers=quant_layers)

        # all layers has the same quantization config
        if 'groupsize' not in quant_config.quant_config_by_op:
            for layer_name in list(layers.keys()):
                if layer_name not in quant_config.quant_config_by_op:
                    del layers[layer_name]
        else:  # removed unquantized layer, TODO load layers from safetensors
            use_heuristic = False
            if not use_heuristic:
                all_keys = _load_check_point(None, model_name_or_path, get_keys_only=True)
                all_keys = [i.replace('.qweight', '') for i in all_keys if i.endswith('.qweight')]
                for layer_name in set(layers.keys()) - set(all_keys):
                    del layers[layer_name]
            else:
                for layer_name in list(layers.keys()):
                    if len(layer_name.split('.')) <= 3:
                        del layers[layer_name]

        target_layer = utils.modelutils.select_quant_linear(
            quant_config.version, quant_config.bits(), quant_config.method)
        torch.set_default_device("cuda")
        utils.modelutils.make_mixbits_quant_linear(
            model, layers, quant_config.quant_config_by_op, target_layer=target_layer)
        torch.set_default_device("cpu")
        if quant_config.method == "awq":
            from ..quantization.quant_awq import scale_activations
            scale_activations(model)
        del layers
        # if low_cpu_mem_usage:
        #    model = model.cuda()
        model.tie_weights()  # works with init_empty_weights and load_checkpoint_and_dispatch
        try:
            # bias issue
            no_split_module_classes = get_no_split_layer_type_name(model)
            assert no_split_module_classes is None
            if torch.cuda.mem_get_info()[1]/1024/1024/1024 < 8:
                model = accelerate.big_modeling.load_checkpoint_and_dispatch(
                    model,
                    checkpoint=_get_resolved_weight_or_index_file(model_name_or_path, quant_config),
                    device_map=device_map,
                    no_split_module_classes=no_split_module_classes,
                    dtype=torch_dtype,
                )
            else:
                raise Exception("")
        except Exception:
            clear_memory()
            all_missing_keys, all_unexpected_keys = _load_check_point(model, model_name_or_path)
            all_unexpected_keys = [i for i in all_unexpected_keys if not i.endswith('.bias')]
            if len(all_unexpected_keys) != 0:
                logger.warn(f"Unexpected keys in checkpoint: {all_unexpected_keys}")

        # autogptq has extra -1 in qzeros but we don't have it.
        if quant_config.COMPATIBLE_WITH_AUTOGPTQ:
            qlayers = utils.find_layers(model, [target_layer])
            for _, qlayer in tqdm.tqdm(qlayers.items(), desc="Repacking AutoGPTQ qzeros..."):
                qlayer.handle_qzeros_for_autogptq()
            import os
            os.environ["COMPATIBLE_WITH_AUTOGPTQ"] = '0'
            # The kernel with COMPATIBLE_WITH_AUTOGPTQ has some bugs now in XbitOps, let's always replace qzeros

        return model

    @staticmethod
    def save_pretrained(model, tokenizer, save_directory: Union[str, Path], pack_mode: str, repack_func, save_serialization: bool = False):
        quant_config_by_layer, quant_config = model.quant_config_by_layer, model.quant_config
        if pack_mode != quant_config.version and pack_mode != "AUTO":
            repack_func()
        model.config.quantization_config = quant_config
        model.save_pretrained(save_directory, save_serialization=save_serialization)
        tokenizer is not None and tokenizer.save_pretrained(save_directory)

        with open(save_directory + "/quant_config_by_layer.json", 'w') as fp:
            fp.write(json.dumps(quant_config_by_layer))
        with open(save_directory + "/quantize_config.json", 'w') as fp:
            fp.write(json.dumps(quant_config.to_dict()))
