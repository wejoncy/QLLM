from .quant_awq import AWQQuant
from .quant_gptq import GPTQQuant
from .quant_hqq import HQQQuant
from .config_builder import build_config

def get_quantizer(config):
    if config.method == "gptq":
        return GPTQQuant(config)
    elif config.method == "awq":
        return AWQQuant(config)
    elif config.method == "hqq":
        return HQQQuant(config)
    else:
        raise NotImplementedError
