import torch
from torch import nn
from ..utils import comm_utils


class QuantFrameBase:
    def __init__(self, args) -> None:
        self.rec_use_cache = False
        self.quant_layers = [torch.nn.Linear]
        self.args = args
        self.swap_device = torch.device('cpu')

    @torch.no_grad()
    def prepare(self, model):
        print('Starting ...')
        self.rec_use_cache = model.config.use_cache

        model = model.cpu()
        model.config.use_cache = False
        return model

    @torch.no_grad()
    def extract_prefix(self, model):
        state_dict_prefix = None
        for name in model.state_dict().keys():
            if '.0.' not in name:
                continue
            state_dict_prefix = name.split('.0.')[0]
        assert state_dict_prefix is not None, "state_dict_prefix not found"
        return state_dict_prefix

    @torch.no_grad()
    def extract_layers(self, model):
        attention_layers = None
        pre_layers_of_attention = []  # enmbedding layer, norm layer

        # find the attention layers, and the pre layers of attention layers
        # A layer-block has more than 32 layers usually
        transformer_model = model
        while len(list(transformer_model.named_children())) <= 2:
            decoder_name = next(transformer_model.named_children())[0]
            transformer_model = getattr(transformer_model, decoder_name)
        for name, layer in transformer_model.named_children():
            if type(layer) in [torch.nn.ModuleList]:
                attention_layers = layer
                break
            else:
                pre_layers_of_attention.append(layer)
        return attention_layers, pre_layers_of_attention

    def hijack_block_inputs(self, model, dataloader, args, dev):
        dtype = next(iter(model.parameters())).dtype
        # torch.zeros((args.nsamples, dataloader[0][0].shape[-1], model.config.d_model), dtype=dtype)
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

        attention_layers, pre_layers_of_attention = self.extract_layers(model)
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
        if str(layer_id+1) in args.mix_qlayer_conf:
            layer_key = str(layer_id+1)
            args.wbits = args.mix_qlayer_conf[layer_key].get('wbits', args.wbits)
            args.groupsize = args.mix_qlayer_conf[layer_key].get('groupsize', args.groupsize)

    def quantize(self, model, dataloader, args, dev):
        raise NotImplementedError
