import torch
import torch.nn as nn
import tqdm
from texttable import Texttable

from .quant_frame_base import QuantFrameBase
from .gptq import GPTQ, Observer
from ..utils import find_layers, gen_conditions
from ..utils.logger import get_logger
from . import sequential_layes_config
logger = get_logger('qllm')

class ObserverHelper:
    def __init__(self, args) -> None:
        self.observer = Observer()
        self.args = args

    def submit(self, name, layerid, gptq, error):
        if self.args.observe:
            self.observer.submit(name=name, layerid=layerid, gptq=gptq, error=error)
            return True
        return False

    def post_quant(self, quantizers, state_dict_prefix):
        if not self.args.observe:
            return
        args = self.args
        logger.debug(self.observer.print())
        conditions = gen_conditions(args.wbits, args.groupsize)
        for item in tqdm.tqdm(self.observer.items(), desc="Optimizing with mix bits/groupsize"):
            name = item[0]
            layerid = item[1]
            gptq = item[2]['gptq']
            error = item[2]['error']
            target = error / 2

            table = Texttable()
            table.header(['wbits', 'groupsize', 'error'])
            table.set_cols_dtype(['i', 'i', 'f'])
            table.add_row([args.wbits, args.groupsize, error])

            logger.debug('Optimizing {} {} ..'.format(name, layerid))
            for wbits, groupsize in conditions:

                if error < target:
                    # if error dropped 50%, skip
                    break

                gptq.quantizer.configure(wbits, perchannel=True, sym=args.sym, mse=False)

                scale, zero, g_idx, error = gptq.fasterquant(
                    percdamp=args.percdamp, groupsize=groupsize, actorder=args.act_order, name=name)

                table.add_row([wbits, groupsize, error])
                quantizers[f'{state_dict_prefix}.{layerid}.{name}'] = (
                    gptq.quantizer.cpu(), scale.cpu(), zero.cpu(), g_idx.cpu(), wbits, groupsize)

            logger.debug(table.draw())
            logger.debug('\n')
            gptq.layer.to('cpu')
            gptq.free()


class GPTQQuant(QuantFrameBase):
    def __init__(self, args) -> None:
        super().__init__(args)

    def hijack_internal_block(self, gptq, subset, layer_block, inps, layer_kwargs):
        dev = next(layer_block.parameters()).device

        def add_batch(name):
            def tmp(_, inp, out):
                gptq[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in subset:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(len(inps)):
            _ = layer_block(inps[j].unsqueeze(0).to(dev), **layer_kwargs)
        for h in handles:
            h.remove()

    def quantize(self, model, dataloader, dev):
        args = self.args
        model = self.prepare(model)
        state_dict_prefix = self.extract_prefix(model)
        inps, outs, attention_layers, layer_input_args = self.hijack_block_inputs(model, dataloader, args, dev)
        print('Ready.')

        quantizers = {}
        observer_helper = ObserverHelper(args)
        for i in tqdm.tqdm(range(len(attention_layers)), desc="running GPTQ"):
            self.hook_before_qlayer(i, args)

            block_layer = attention_layers[i].to(dev)
            named_linear_layers = find_layers(block_layer, self.quant_layers)

            # [ TODO ] how to filter out the layers, which won't be quantized or harness the quality
            sequential = [list(named_linear_layers.keys())]
            if args.true_sequential:
                sequential = sequential_layes_config.auto_detect_sequential_layers(
                    sequential, model.__class__.__name__)
            for names in sequential:
                subset = {n: named_linear_layers[n] for n in names}
                gptq = {}
                for name in subset:
                    gptq[name] = GPTQ(subset[name], observe=args.observe)
                    gptq[name].quantizer.configure(args.wbits, perchannel=True, sym=args.sym, mse=False)

                self.hijack_internal_block(gptq, subset, block_layer, inps, layer_input_args)

                for name in subset:
                    scale, zero, g_idx, error = gptq[name].fasterquant(
                        percdamp=args.percdamp, groupsize=args.groupsize, actorder=args.act_order, name=name)
                    quantizers[f'{state_dict_prefix}.{i}.{name}'] = (
                        gptq[name].quantizer.cpu(), scale.cpu(), zero.cpu(), g_idx.cpu(), args.wbits, args.groupsize)

                    if not observer_helper.submit(name=name, layerid=i, gptq=gptq[name], error=error):
                        gptq[name].free()

            # [ TODO ]
            # I am supposing layer's weight should be quantized and modified, we are statisting the error
            # accumulated from the previous layers and compensate next layer
            for j in range(len(dataloader)):
                outs[j] = block_layer(inps[j].unsqueeze(0).to(dev), **layer_input_args)[0].cpu()

            attention_layers[i] = block_layer.cpu()
            del block_layer
            del gptq
            torch.cuda.empty_cache()

            inps, outs = outs, inps

        observer_helper.post_quant(quantizers, state_dict_prefix)

        model.config.use_cache = self.rec_use_cache
        return quantizers
