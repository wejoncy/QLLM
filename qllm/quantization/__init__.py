from .awq_quant import AWQQuant
from .gptq_quant import GPTQQuant


def get_quantizer(args):
    if args.method == 'gptq':
        return GPTQQuant(args)
    elif args.method == 'awq':
        return AWQQuant(args)
    else:
        raise NotImplementedError
