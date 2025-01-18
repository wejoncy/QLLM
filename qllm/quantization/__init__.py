from .awq.quant_awq import AWQQuant
from .gptq.quant_gptq import GPTQQuant
from .hqq.quant_hqq import HQQQuant
from .vptq.quant_vptq import VPTQQuant
from .config_builder import build_config

def get_quantizer(config):
    if config.quant_method == "gptq":
        return GPTQQuant(config)
    elif config.quant_method == "awq":
        return AWQQuant(config)
    elif config.quant_method == "hqq":
        return HQQQuant(config)
    elif config.quant_method == "vptq":
        return VPTQQuant(config)
    else:
        raise NotImplementedError
