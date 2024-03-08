import torch
import tqdm
from texttable import Texttable

from .quant_frame_base import QuantFrameBase
from .gptq import GPTQ, Observer
from ..utils import find_layers, gen_conditions
from ..utils.logger import get_logger
from . import sequential_layes_gptq_config
logger = get_logger('qllm')

class ObserverHelper:
    def __init__(self, config) -> None:
        self.observer = Observer()
        self.quant_config = config

    def submit(self, name, layerid, gptq, error):
        if self.quant_config.allow_mix_bits:
            self.observer.submit(name=name, layerid=layerid, gptq=gptq, error=error)
            return True
        return False

    def post_quant(self, quantizers, state_dict_prefix):
        if not self.quant_config.allow_mix_bits:
            return
        logger.debug(self.observer.print())
        conditions = gen_conditions(self.quant_config.bits, self.quant_config.group_size)
        for item in tqdm.tqdm(self.observer.items(), desc="Optimizing with mix bits/groupsize"):
            name = item[0]
            layerid = item[1]
            gptq = item[2]['gptq']
            error = item[2]['error']
            target = error / 2

            table = Texttable()
            table.header(['wbits', 'groupsize', 'error'])
            table.set_cols_dtype(['i', 'i', 'f'])
            table.add_row([self.quant_config.bits, self.quant_config.group_size, error])

            logger.debug('Optimizing {} {} ..'.format(name, layerid))
            for wbits, groupsize in conditions:

                if error < target:
                    # if error dropped 50%, skip
                    break

                gptq.quantizer.configure(wbits, perchannel=True, sym=self.quant_config.sym, mse=False)

                scale, zero, g_idx, error = gptq.fasterquant(
                    percdamp=self.quant_config.percdamp,
                    groupsize=groupsize,
                    actorder=self.quant_config.desc_act,
                    static_groups=self.quant_config.static_groups,
                    name=name,
                )

                table.add_row([wbits, groupsize, error])
                quantizers[f'{state_dict_prefix}.{layerid}.{name}'] = (
                    gptq.quantizer.cpu(), scale.cpu(), zero.cpu(), g_idx.cpu(), wbits, groupsize)

            logger.debug(table.draw())
            logger.debug('\n')
            gptq.layer.to('cpu')
            gptq.free()


class GPTQQuant(QuantFrameBase):

    def __init__(self, config) -> None:
        super().__init__()
        self.quant_config = config

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

    @torch.inference_mode()
    def do_quantize(self, model, dataloader, model_prefix, dev):
        inps, outs, attention_layers, layer_input_args = self.hijack_block_inputs(model, dataloader, model_prefix, dev)
        print('Ready.')

        quantizers = {}
        observer_helper = ObserverHelper(self.quant_config)
        for i in tqdm.tqdm(range(len(attention_layers)), desc="running GPTQ"):
            self.hook_before_qlayer(i, self.quant_config)

            block_layer = attention_layers[i].to(dev)
            named_linear_layers = find_layers(block_layer, self.quant_layers)

            sequential = [list(named_linear_layers.keys())]
            # filter out the layers that shouldnt be quantized
            true_sequential = sequential_layes_gptq_config.auto_detect_sequential_layers(
                sequential, model.__class__.__name__)
            if not self.quant_config.true_sequential:
                sequential_tmp = [sum(true_sequential, [])]
                if len(sequential_tmp[0]) != len(sequential[0]):
                    # if true_sequential is not the same as sequential, we need to re-order the layers
                    sequential[0] = sorted(sequential_tmp[0], key=lambda x: sequential[0].index(x))
            for names in sequential:
                subset = {n: named_linear_layers[n] for n in names}
                gptq = {}
                for name in subset:
                    gptq[name] = GPTQ(subset[name], allow_mix_bits=self.quant_config.allow_mix_bits)
                    gptq[name].quantizer.configure(
                        self.quant_config.bits, perchannel=True, sym=self.quant_config.sym, mse=False
                    )

                self.hijack_internal_block(gptq, subset, block_layer, inps, layer_input_args)

                for name in subset:
                    scale, zero, g_idx, error = gptq[name].fasterquant(
                        percdamp=self.quant_config.damp_percent,
                        groupsize=self.quant_config.group_size,
                        actorder=self.quant_config.desc_act,
                        static_groups=self.quant_config.static_groups,
                        name=name,
                    )
                    quantizers[f"{model_prefix}.{i}.{name}"] = (
                        gptq[name].quantizer.cpu(),
                        scale.cpu(),
                        zero.cpu(),
                        g_idx.cpu(),
                        self.quant_config.bits,
                        self.quant_config.group_size,
                    )

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

        observer_helper.post_quant(quantizers, model_prefix)
        return quantizers
