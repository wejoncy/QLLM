import torch
import tqdm

from .quant_frame_base import QuantFrameBase
from ._hqq_quantizer import InternalHQQQuantizer
from ..utils import find_layers
from ..utils.logger import get_logger
logger = get_logger('qllm')

class HQQQuant(QuantFrameBase):
    def __init__(self, args) -> None:
        super().__init__(args)
        self.quant_config = dict(group_size=args.groupsize,
                                 bits=args.wbits, method='hqq')

    
    @torch.inference_mode()
    def do_quantize(self, model, dataloader, model_prefix, dev):
        args = self.args
        dataloader = []
        _,_, attention_layers, layer_input_args = self.hijack_block_inputs(model, dataloader, model_prefix, dev)
        print('Ready.')

        quantizers = {}
        for i in tqdm.tqdm(range(len(attention_layers)), desc="running HQQ"):
            block_layer = attention_layers[i].to(dev)
            named_linear_layers = find_layers(block_layer, self.quant_layers)

            # [ TODO ] how to filter out the layers, which won't be quantized or harness the quality
            sequential = [list(named_linear_layers.keys())]
            for names in sequential:
                subset = {n: named_linear_layers[n] for n in names}
                gptq = {}
                for name in subset:
                    gptq[name] = InternalHQQQuantizer(subset[name])
                    gptq[name].configure(args.wbits, channel_wise=True, group_size=args.groupsize, 
                                         optimize=True, round_zero=True, axis=1)
                    scale, zero = gptq[name].quantize()
                    quantizers[f'{model_prefix}.{i}.{name}'] = (
                        gptq[name], scale.cpu(), zero.cpu(), None, args.wbits, args.groupsize)

                    gptq[name].free()


            attention_layers[i] = block_layer.cpu()
            del block_layer
            del gptq
            torch.cuda.empty_cache()

        return quantizers

