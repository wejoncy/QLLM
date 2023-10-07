from texttable import Texttable
import os
#if "CUDA_VISIBLE_DEVICES" not in os.environ: # NOQA
#    os.environ["CUDA_VISIBLE_DEVICES"] = "1" # NOQA

import torch.nn as nn
import torch
import numpy as np

import argparse
import time
from pathlib import Path
import json
import sys
import contextlib

from .utils import get_loaders, disable_huggingface_init
from .utils import find_layers, DEV,export_quant_table, gen_conditions
from .gptq import GPTQ, Observer
from .quant import make_mixbits_quant_linear,QuantLinear

NEED_CHECK_PACK = False


class ModelQuantizationBase(object):
    def __init__(self) -> None:
        super().__init__()
        self.quant_layers = [torch.nn.Linear]
    
    def get_torch_model(self, args, dev='cpu'):
        print(f"loading model from {args.model}")
        disable_huggingface_init()
        from transformers import AutoModelForCausalLM
        llm = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, trust_remote_code=True).to(dev)
        from pathlib import Path
        cache_dir = Path(f"/tmp/qllm_v1/{args.model.replace(' ','_')}_{args.dataset}_dataloader.pt")
        cache_dir.parent.mkdir(parents=True, exist_ok=True)
        print(f"loading dataset from {args.dataset}")
        if cache_dir.exists():
            print(f"found cached dataloader in {cache_dir}")
            dataloader = torch.load(cache_dir)
        else:
            dataloader, _ = get_loaders(args.dataset, nsamples=args.nsamples, 
                seed=args.seed, model=args.tokenizer, seqlen=2048)
            torch.save(dataloader, str(cache_dir))
        return llm,dataloader # model, dataloader

    def __load_quant(self, qmodel, args):
        print(f"loading quantized model from {qmodel}")
        args.model = args.load

        @contextlib.contextmanager
        def stack_attr(attrs:list):
            old_attr = []
            new_method = lambda x, *args, **kwargs: x
            for attr in attrs:
                old_attr.append(getattr(torch.Tensor, attr))
                setattr(torch.Tensor, attr, new_method)
            yield
            for idx,attr in enumerate(attrs):
                setattr(torch.Tensor, attr, old_attr[idx])

        with stack_attr(['normal_','uniform_','kaiming_uniform_','kaiming_normal_']):
            model, dataloader = self.get_torch_model(args, dev='cpu')
        import transformers
        layers = find_layers(model, layers=self.quant_layers)
        # backward compatability
        if not (Path(qmodel)/"quant.op.json").exists():
            quant_layers_json = {layer_name: {"groupsize": args.groupsize, "wbits": args.wbits}
                                for layer_name in layers.keys() if len(layer_name.split('.')) > 3}
            open(Path(qmodel)/"quant.op.json").write(json.dumps(quant_layers_json))

        # load quant info
        with open(Path(qmodel)/"quant.op.json") as fp:
            qunat_info = json.load(fp)
        for layer_name in list(layers.keys()):
            if layer_name not in qunat_info:
                del layers[layer_name]

        make_mixbits_quant_linear(model, layers, qunat_info)
        del layers
        import glob
        weight_bins = glob.glob(os.path.abspath(qmodel)+'/pytorch_model*.bin')
        weight_dict = torch.load(weight_bins[0])
        for i in range(1, len(weight_bins)):
            weight_dict.update(torch.load(weight_bins[i]))
        model.load_state_dict(weight_dict)
        # quant.autotune_warmup_linear(model, transpose=False)
        return model, dataloader

    # you shouldn't rewrite this function
    @torch.no_grad()
    def __quant_by_sequential(self, model, dataloader, args, dev):
        print('Starting ...')
        model = model.cpu()
        use_cache = model.config.use_cache
        model.config.use_cache = False

        attention_layers = None
        pre_layers_of_attention = [] # enmbedding layer, norm layer

        transformer_model = model
        while len(list(transformer_model.named_children())) <=2:
            decoder_name = next(transformer_model.named_children())[0]
            transformer_model = getattr(transformer_model, decoder_name)
        for name,layer in transformer_model.named_children():
            if type(layer) in [torch.nn.ModuleList]:
                attention_layers = layer
                break
            else:
                pre_layers_of_attention.append(layer)

        state_dict_prefix = None
        for name in model.state_dict().keys():
            if '.0.' not in name:
                continue
            state_dict_prefix = name.split('.0.')[0]
        assert state_dict_prefix is not None, "state_dict_prefix not found"

        for layer in pre_layers_of_attention:
            layer = layer.to(dev)
        attention_layers[0] = attention_layers[0].to(dev)

        dtype = next(iter(model.parameters())).dtype
        swap_device = torch.device('cpu')
        #torch.zeros((args.nsamples, dataloader[0][0].shape[-1], model.config.d_model), dtype=dtype)
        inps = [0 for i in range(len(dataloader))]
        cache = {'i': 0, 'attention_mask': None}

        class Catcher(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module

            def forward(self, inp, **kwargs):
                inps[cache['i']] = inp.to(swap_device)
                cache['i'] += 1
                cache['attention_mask'] = kwargs['attention_mask']
                if 'position_ids' in kwargs:
                    cache['position_ids'] = kwargs['position_ids']
                if 'alibi' in kwargs:
                    cache['alibi'] = kwargs['alibi']
                raise ValueError

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
        torch.cuda.empty_cache()

        inps = torch.cat(inps, dim=0)
        outs = torch.zeros_like(inps)
        attention_mask = cache['attention_mask']
        layer_input_args = {'attention_mask': cache['attention_mask']}
        if 'position_ids' in cache:
            layer_input_args['position_ids'] = cache['position_ids']
        if 'alibi' in cache:
            layer_input_args['alibi'] = cache['alibi']

        print('Ready.')

        quantizers = {}
        observer = Observer()
        for i in range(len(attention_layers)):
            if str(i+1) in args.mix_qlayer_conf:
                layer_key = str(i+1)
                args.wbits = args.mix_qlayer_conf[layer_key].get('wbits', args.wbits)
                args.groupsize = args.mix_qlayer_conf[layer_key].get('groupsize', args.groupsize)

            print(f'Quantizing layer {i+1}/{len(attention_layers)}..')
            print('+------------------+--------------+------------+-----------+-------+')
            print('|       name       | weight_error | fp_inp_SNR | q_inp_SNR | time  |')
            print('+==================+==============+============+===========+=======+')

            layer = attention_layers[i].to(dev)
            full = find_layers(layer, self.quant_layers)
            if args.true_sequential:
                sequential = [['self_attn.k_proj', 'self_attn.v_proj', 'self_attn.q_proj'], [
                    'self_attn.o_proj'], ['mlp.up_proj', 'mlp.gate_proj'], ['mlp.down_proj']]
            else:
                sequential = [list(full.keys())]

            for names in sequential:
                subset = {n: full[n] for n in names}
                gptq = {}
                for name in subset:
                    gptq[name] = GPTQ(subset[name], observe=args.observe)
                    gptq[name].quantizer.configure(args.wbits, perchannel=True, sym=args.sym, mse=False)

                def add_batch(name):
                    def tmp(_, inp, out):
                        gptq[name].add_batch(inp[0].data, out.data)

                    return tmp

                handles = []
                for name in subset:
                    handles.append(subset[name].register_forward_hook(add_batch(name)))
                for j in range(len(dataloader)):
                    outs[j] = layer(inps[j].unsqueeze(0).to(dev), **layer_input_args)[0].to(swap_device)
                for h in handles:
                    h.remove()

                for name in subset:
                    scale, zero, g_idx, error = gptq[name].fasterquant(
                        percdamp=args.percdamp, groupsize=args.groupsize, actorder=args.act_order, name=name)
                    quantizers[f'{state_dict_prefix}.{i}.{name}'] = (
                        gptq[name].quantizer.cpu(), scale.cpu(), zero.cpu(), g_idx.cpu(), args.wbits, args.groupsize)

                    if args.observe:
                        observer.submit(name=name, layerid=i, gptq=gptq[name], error=error)
                    else:
                        gptq[name].free()

            # [ TODO ]
            # I am supposing layer's weight should be quantized and modified, we are statisting the error
            # accumulated from the previous layers and compensate next layer
            for j in range(len(dataloader)):
                outs[j] = layer(inps[j].unsqueeze(0).to(dev), **layer_input_args)[0].cpu()

            attention_layers[i] = layer.cpu()
            del layer
            del gptq
            torch.cuda.empty_cache()

            inps, outs = outs, inps
            print('+------------------+--------------+------------+-----------+-------+')
            print('\n')

        if args.observe:
            observer.print()
            conditions = gen_conditions(args.wbits, args.groupsize)
            for item in observer.items():
                name = item[0]
                layerid = item[1]
                gptq = item[2]['gptq']
                error = item[2]['error']
                target = error / 2

                table = Texttable()
                table.header(['wbits', 'groupsize', 'error'])
                table.set_cols_dtype(['i', 'i', 'f'])
                table.add_row([args.wbits, args.groupsize, error])

                print('Optimizing {} {} ..'.format(name, layerid))
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

                print(table.draw())
                print('\n')
                gptq.layer.to('cpu')
                gptq.free()

        model.config.use_cache = use_cache
        return quantizers


    @torch.no_grad()
    def eval_model(self, model, dev):
        print('Evaluating ...')
        print("you should rewrite this function for your model")

    # TODO: perform packing on GPU
    def pack_model(self, model, quantizers):
        attention_layers = find_layers(model, self.quant_layers)
        attention_layers = {n: attention_layers[n] for n in quantizers}
        quant_info = {key: {"wbits": value[-2], "groupsize": value[-1]} for key, value in quantizers.items()}
        make_mixbits_quant_linear(model, quantizers, quant_info)
        qlayers = find_layers(model, [QuantLinear])
        print('Packing ...')
        for name in qlayers:
            print(name)
            quantizers[name], scale, zero, g_idx, _, _ = quantizers[name]
            # rewrite weight as quantized
            if NEED_CHECK_PACK:
                qlayers[name].oweight = qlayers[name].weight_qdq(attention_layers[name], scale, zero, g_idx).cuda()
                attention_layers[name].weight.data = qlayers[name].oweight
                assert (qlayers[name].oweight == qlayers[name].weight_qdq(attention_layers[name], scale, zero, g_idx).cuda()).all()
                attention_layers[name].nbits = qlayers[name].bits

            qlayers[name].pack_gpu(attention_layers[name], scale, zero, g_idx)

        # quant.make_linear_qdq_back(model,attention_layers)
        # quant.autotune_warmup_linear(model, transpose=False)

        print('Done.')
        return model, quant_info

    def pipeline_to_multiple_gpu(self, model, gpulist:list, sample_inputs):
        def input_gpu_device_hook(mod, inputs):
            modifyed_inputs = []
            first_dev = None
            for layer_input in inputs:
                if type(layer_input) is not torch.Tensor:
                    modifyed_inputs.append(layer_input)
                elif hasattr(mod, 'weight'):
                    modifyed_inputs.append(layer_input.to(mod.weight.device))
                elif hasattr(next(mod.children(), None), 'weight'):
                    modifyed_inputs.append(layer_input.to(next(mod.children()).weight.device))
                elif first_dev is not None and layer_input.device != first_dev:
                    modifyed_inputs.append(layer_input.to(first_dev))
                else:
                    modifyed_inputs.append(layer_input)
                if first_dev is None:
                    first_dev = modifyed_inputs[0].device
            return tuple(modifyed_inputs)
        
        def move_layer_to_device_rurc(mod,dev):
            mod.to(dev)
            for layer in mod.named_children():
                move_layer_to_device_rurc(layer[1],dev)

        #model.register_module_forward_pre_hook(input_gpu_device_hook)
        model.register_forward_pre_hook(input_gpu_device_hook)
        pre_fix = list(model.named_children())[0][0]
        for name,module in model.get_decoder().named_children(): 
            module.register_forward_pre_hook(input_gpu_device_hook)           
            if type(module) in [torch.nn.ModuleList]:
                import math
                num_layers_on_each_gpu = math.floor(len(module)/len(gpulist))
                for idx,attn_layer in enumerate(module):
                    attn_layer.register_forward_pre_hook(input_gpu_device_hook)           

                    to_dev = gpulist[min(idx//num_layers_on_each_gpu,2)]
                    attn_layer.to(to_dev)
                    move_layer_to_device_rurc(attn_layer, to_dev)
                    print(f"move {pre_fix}.{name}.{idx} to {to_dev}")
            else:
                module.to(gpulist[0])
                print(f"move {pre_fix}.{name} to {gpulist[0]}")
        with torch.no_grad():
            out=model(sample_inputs[0], attention_mask=sample_inputs[1].cuda(0))
        print(out)
        return model



    def export_onnx(self, model, onnx_path, sample_inputs: tuple):
        #model = model.cpu().float()
        model=model.cuda()
        from pathlib import Path
        import shutil
        onnx_path = Path(onnx_path).absolute()
        assert onnx_path.suffix == '.onnx'
        inputs = {'input_ids': sample_inputs[0].to(model.device), "attention_mask": sample_inputs[1].to(model.device)}
        onnx_filepath_export_multi_files_tmp = onnx_path.parent/'tmp/tmp.onnx'
        onnx_filepath_export_multi_files_tmp.parent.exists() and shutil.rmtree(onnx_filepath_export_multi_files_tmp.parent)
        os.makedirs(onnx_filepath_export_multi_files_tmp.parent)

        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        past_key_values = None
        onnx_inputs = (input_ids, past_key_values, attention_mask, None, None, None, True, False, False, False)
        onnx_inp_names = ("input_ids", "attention_mask")
        onnx_out_names = ("logits",)
        onnx_dynamic_axes = {"input_ids": {0: 'batch_size', 1: "seq_len"},
                            "attention_mask": {0: 'batch_size', 1: "seq_len"}}
        torch.onnx.export(model=model, args=onnx_inputs, f=str(onnx_filepath_export_multi_files_tmp), verbose=False, opset_version=16,
                        input_names=onnx_inp_names, output_names=onnx_out_names, dynamic_axes=onnx_dynamic_axes)
        import onnx
        onnx_model = onnx.load(str(onnx_filepath_export_multi_files_tmp))

        onnx_path.exists() and onnx_path.unlink()
        (onnx_path.parent/'mpt_ext.data').exists() and (onnx_path.parent/'mpt_ext.data').unlink()
        onnx.save_model(onnx_model, str(onnx_path), save_as_external_data=True, all_tensors_to_one_file=True,
                        location="mpt_ext.data", size_threshold=1024, convert_attribute=False)

    def append_default_args(self):
        if '--wbits' not in sys.argv:
            sys.argv += ['--wbits', '4']

        if '--groupsize' not in sys.argv:
            sys.argv += ['--groupsize', '128']

        if '--nsamples' not in sys.argv:
            sys.argv += ['--nsamples', '512']

        #if '--export_onnx' not in sys.argv:
        #    sys.argv += ['--export_onnx', './mpt_onnx_q4/mpt.onnx']
    #
        #if '--eval' not in sys.argv:
        #    sys.argv += ['--eval']

        #if '--save' not in sys.argv:
        #    sys.argv += ['--save', './mpt_q4']
        #if '--load' not in sys.argv:
        #    sys.argv += ['--load', './mpt_q4']

    def define_basic_args(self):
        #,'--observe','--act-order'
        self.append_default_args()
        parser = argparse.ArgumentParser()

        parser.add_argument('--model', type=str, default="", help='float/float16 model to load, such as [mosaicml/mpt-7b]')
        parser.add_argument('--tokenizer', type=str, default="", help='default same as [model]')
        parser.add_argument('--dataset', type=str, default='c4',
                            choices=['wikitext2', 'ptb', 'c4'], help='Where to extract calibration data from.')
        parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
        parser.add_argument('--nsamples', type=int, default=128, help='Number of calibration data samples.')
        parser.add_argument('--percdamp', type=float, default=.01,
                            help='Percent of the average Hessian diagonal to use for dampening.')
        parser.add_argument('--nearest', action='store_true', help='Whether to run the RTN baseline.')
        parser.add_argument('--wbits', type=int, default=16,
                            choices=[2, 3, 4, 5, 6, 7, 8, 16], help='#bits to use for quantization; use 16 for evaluating base model.')
        parser.add_argument('--trits', action='store_true', help='Whether to use trits for quantization.')
        parser.add_argument('--mix_qlayer_conf', type=str, default=None, help='Mix quantization layer configuration.(groupsize,wbits)')
        parser.add_argument('--groupsize', type=int, default=-1,
                            help='Groupsize to use for quantization; default uses full row.')
        parser.add_argument('--eval', action='store_true', help='evaluate quantized model.')
        parser.add_argument('--save', type=str, default='', help='Save quantized checkpoint under this name.')
        parser.add_argument('--save_safetensors', type=str, default='',
                            help='Save quantized `.safetensors` checkpoint under this name.')
        parser.add_argument('--load', type=str, default='', help='Load quantized model.')
        parser.add_argument('--check', action='store_true',
                            help='Whether to compute perplexity during benchmarking for verification.')
        parser.add_argument('--sym', action='store_true', help='Whether to perform symmetric quantization.')
        parser.add_argument('--act-order', action='store_true', help='Whether to apply the activation order GPTQ heuristic')
        parser.add_argument('--true-sequential', action='store_true', help='Whether to run in true sequential model.')
        parser.add_argument('--new-eval', action='store_true', help='Whether to use the new PTB and C4 eval')
        parser.add_argument('--layers-dist', type=str, default='',
                            help='Distribution of layers across GPUs. e.g. 2:1:1 for 2 layers on GPU 0, 1 layer on GPU 1, and 1 layer on GPU 2. Any remaining layers will be assigned to your last GPU.')
        parser.add_argument('--observe',
                            action='store_true',
                            help='Auto upgrade layer precision to higher precision, for example int2 to int4, groupsize 128 to 64. \
                When this feature enabled, `--save` or `--save_safetensors` would be disable.')
        parser.add_argument('--quant-directory', type=str, default=None,
                            help='Specify the directory for export quantization parameters to toml format. `None` means no export by default.')
        parser.add_argument('--export_onnx', type=str, default=None, help='where does the onnx model save to.')

        return parser


    def run(self, args):
        if args.tokenizer == "":
            args.tokenizer = args.model
        if args.layers_dist:
            gpu_dist = [int(x) for x in args.layers_dist.split(':')]
        else:
            gpu_dist = []

        if type(args.load) is not str:
            args.load = args.load.as_posix()

        if args.load:
            model, dataloader = self.__load_quant(args.load, args)
            model.eval()
        elif args.model:
            model, dataloader = self.get_torch_model(args, dev='cpu')
            model.eval()
        else:
            raise ValueError("either --model or --load must be specified")

        if not args.load and args.wbits < 16 and not args.nearest:
            if args.mix_qlayer_conf:
                args.mix_qlayer_conf = json.load(open(args.mix_qlayer_conf))
            else:
                args.mix_qlayer_conf = {}
            tick = time.time()
            quantizers = self.__quant_by_sequential(model, dataloader, args, DEV)
            model, quant_info = self.pack_model(model, quantizers)
            print("Finished quantization and packing weight, time cost:", time.time() - tick)

        if args.quant_directory is not None:
            export_quant_table(quantizers, args.quant_directory)

        if not args.observe and args.save:
            model.save_pretrained(args.save)
            open(args.save+"/quant.op.json", 'w').write(json.dumps(quant_info))

        if args.eval:
            self.eval_model(model, DEV)

        if args.export_onnx:
            self.export_onnx(model, args.export_onnx, dataloader[0])

        if not args.observe and args.save_safetensors:
            from safetensors.torch import save_file as safe_save
            state_dict = model.state_dict()
            state_dict = {k: v.clone().contiguous() for k, v in state_dict.items()}
            safe_save(state_dict, args.save_safetensors)


def main():
    print("quantize LLM with base engine")
    model_quanter = ModelQuantizationBase()
    parser = model_quanter.define_basic_args()
    args = parser.parse_args()
    print(args)
    model_quanter.run(args)
    
if __name__ == '__main__':
    main()
