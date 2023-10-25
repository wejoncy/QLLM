from .quant_awq import AWQQuant
from .quant_gptq import GPTQQuant


def get_quantizer(args):
    if args.method == 'gptq':
        return GPTQQuant(args)
    elif args.method == 'awq':
        return AWQQuant(args)
    else:
        raise NotImplementedError
