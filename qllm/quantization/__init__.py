from .quant_awq import AWQQuant
from .quant_gptq import GPTQQuant
from .quant_hqq import HQQQuant

def get_quantizer(args):
    if args.method == 'gptq':
        return GPTQQuant(args)
    elif args.method == 'awq':
        return AWQQuant(args)
    elif args.method == 'hqq':
        return HQQQuant(args)
    else:
        raise NotImplementedError
