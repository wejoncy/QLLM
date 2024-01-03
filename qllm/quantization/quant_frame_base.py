import torch
from torch import nn
from ..utils import comm_utils
from ..utils.logger import get_logger
from ..utils.modelutils import get_op_by_name

logger = get_logger()

class QuantFrameBase:
    def __init__(self, args) -> None:
        self.rec_use_cache = False
        self.quant_layers = [torch.nn.Linear]
        self.args = args
        self.swap_device = torch.device('cpu')

    @torch.no_grad()
    def prepare(self, model):
        print('Starting ...')
        self.rec_use_cache = getattr(model.config, 'use_cache', False)

        model = model.cpu()
        model.config.use_cache = False
        return model

    @torch.no_grad()
    def extract_prefix(self, model):
        '''
        heristicly extract the prefix of the state_dict
        support encoder-decoder model and decoder-only model
        a model usually has a state_dict like this:
        x.embed
        x.model x.encoder x.decoder
        x.lm_head
        x.dropout
        x.loss
        '''

        state_dict_prefix = None
        prefix_list = [] #encoder/decoder
        for name in model.state_dict():
            if '.0.' not in name:
                continue
            state_dict_prefix = name.split('.0.')[0]
            prefix_list.append(state_dict_prefix)

        if len(prefix_list) > 1: # encoder-decoder model
            min_len = min([len(i) for i in prefix_list])
        prefix_list = [i for i in prefix_list if len(i) == min_len]
        prefix_list = set(prefix_list)
        if len(prefix_list) > 1:
            raise ValueError(f"Multiple prefix found: {prefix_list}, encoder-decoder model is not supported")
        assert prefix_list, "state_dict_prefix not found"
        return prefix_list

    @torch.no_grad()
    def extract_layers(self, model, model_prefix):
        attention_layers = None
        pre_layers_of_attention = []  # enmbedding layer, norm layer
        # find the attention layers, and the pre layers of attention layers
        transformer_model = get_op_by_name(model, '.'.join(model_prefix.split('.')[:-1]))
        for _, layer in transformer_model.named_children():
            if type(layer) in [torch.nn.ModuleList]:
                attention_layers = layer
                break
            else:
                pre_layers_of_attention.append(layer)
        assert attention_layers is not None, "attention_layers not found"
        return attention_layers, pre_layers_of_attention

    def hijack_block_inputs(self, model, dataloader, model_prefix, dev):
        inps = []
        layer_input_args = {}
        swap_device = self.swap_device

        class Catcher(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module

            def forward(self, inp, **kwargs):
                inps.append(inp.to(swap_device))
                layer_input_args.update(kwargs)
                raise ValueError

        attention_layers, pre_layers_of_attention = self.extract_layers(model, model_prefix)
        for layer in pre_layers_of_attention:
            layer = layer.to(dev)
        attention_layers[0] = attention_layers[0].to(dev)
        attention_layers[0] = Catcher(attention_layers[0])
        for batch in dataloader:
            try:
                model(batch[0].to(dev))
            except ValueError:
                pass
        attention_layers[0] = attention_layers[0].module
        attention_layers[0] = attention_layers[0].cpu()
        for layer in pre_layers_of_attention:
            layer = layer.cpu()
        comm_utils.clear_memory()

        inps = torch.cat(inps, dim=0)
        outs = torch.zeros_like(inps)
        return inps, outs, attention_layers, layer_input_args

    def hook_before_qlayer(self, layer_id, args):
        if str(layer_id + 1) in args.mix_qlayer_conf:
            layer_key = str(layer_id + 1)
            args.wbits = args.mix_qlayer_conf[layer_key].get('wbits', args.wbits)
            args.groupsize = args.mix_qlayer_conf[layer_key].get('groupsize', args.groupsize)

    def do_quantize(self, model, dataloader, model_prefix, dev):
        raise NotImplementedError

    def quantize(self, model, dataloader, dev):
        model = self.prepare(model)
        quantizers = {}
        state_dict_prefix:list = self.extract_prefix(model)
        for prefix in state_dict_prefix:
            quantizers.update(self.do_quantize(model, dataloader, prefix, dev))

        model.config.use_cache = self.rec_use_cache
        model.quant_config = self.quant_config
        return quantizers
