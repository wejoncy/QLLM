import tqdm
import argparse
import time
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import torch
import torch.nn as nn
import quant

import sys

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.getcwd())
import  loralib as lora

from gptq import GPTQ, Observer
from utils import find_layers, DEV, set_seed, get_wikitext2, get_ptb, get_c4, get_ptb_new, get_c4_new, get_loaders, export_quant_table, gen_conditions
from texttable import Texttable
DO_QDQ = False


def get_mpt(model, argv_user, nsamples, is_load_quant=False):
    if is_load_quant or argv_user[argv_user.index('--model_name_or_path')+1] not in ['ckpt/mpt-7b-storywriter/', 'ckpt/mpt-7b-storywriter']:
        lora_ind = argv_user.index('--use_lora')
        argv_user[lora_ind+1] = 'False'


    import examples_ads
    if 'mpt-' in ' '.join(argv_user):
        from examples_ads import run_mpt_prompt as prompt_script
    else:
        from examples_ads import run_llama_prompt as prompt_script
    argv_user.insert(0, prompt_script.__file__)
    argv_back = sys.argv
    sys.argv = argv_user

    print('\n\n\nCalling custom model with args ... \n', sys.argv)
    model, data_sets = prompt_script.main(True)
    new_data = []
    for idx, indata in enumerate(data_sets):
        if idx>=nsamples:break
        input_ = (torch.tensor([indata["input_ids"]]), torch.tensor([indata["attention_mask"]]))            
        new_data.append(input_)
    return model.half(),new_data


def load_quant(qmodel, argv_user, args):
    argv_user[argv_user.index('--model_name_or_path')+1] = os.path.abspath(qmodel)
    os.environ["nbits"] = "4"

    model,dataloader = get_mpt(args.model, argv_user, args.nsamples, is_load_quant=True)
    layers = find_layers(model, layers=[torch.nn.Linear])
    for layer_name in list(layers.keys()):
        if len(layer_name.split('.')) <= 3:
            del layers[layer_name]
    quant.replace_quant_linear_layer(model,layers ,args.wbits, args.groupsize)
    model_name_or_path = qmodel
    import glob
    from pathlib import Path
    weight_bins = glob.glob(
        str(Path(model_name_or_path).absolute()/'pytorch_model*.bin'))
    assert len(weight_bins) > 0, 'No weight bin found.'
    for i in tqdm.tqdm(range(len(weight_bins)), desc="loading weights"):
        model.load_state_dict(torch.load(weight_bins[i]), strict=False)
    #quant.autotune_warmup_linear(model, transpose=False)
    return model.cuda(),dataloader

@torch.no_grad()
def mpt_sequential(model, dataloader, dev):
    print('Starting ...')
    model=model.cpu()
    use_cache = model.config.use_cache
    model.config.use_cache = False

    is_mpt = hasattr(model, 'get_decoder')
    d_model = 4096
    if is_mpt:
        inter_decoder = model.get_decoder()
        layers = model.get_decoder().blocks
        inter_decoder.wte = inter_decoder.wte.to(dev)
        inter_decoder.norm_f = inter_decoder.norm_f.to(dev)
        d_model = model.config.d_model
        prefix = 'transformer.blocks'
    else:
        inter_decoder = model.model
        layers = model.model.layers
        inter_decoder.embed_tokens = inter_decoder.embed_tokens.to(dev)
        inter_decoder.norm = inter_decoder.norm.to(dev)
        d_model = model.config.hidden_size
        prefix = 'model.layers'


    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    #model.config.max_seq_len

    inps = torch.zeros((args.nsamples, dataloader[0][0].shape[-1], d_model), dtype=dtype, device=dev)
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):

        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            if 'position_ids' in kwargs:
                cache['position_ids'] = kwargs['position_ids']
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    if is_mpt:
        inter_decoder.wte = inter_decoder.wte.cpu()
        inter_decoder.norm_f = inter_decoder.norm_f.cpu()
    else:
        inter_decoder.embed_tokens = inter_decoder.embed_tokens.cpu()
        inter_decoder.norm = inter_decoder.norm.cpu()
        # higher precision with true_sequential
        args.true_sequential = True
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    input_args = {'attention_mask': cache['attention_mask']}
    if 'position_ids' in cache:
        input_args['position_ids'] = cache['position_ids']

    print('Ready.')

    quantizers = {}
    observer = Observer()
    for i in range(len(layers)):

        print(f'Quantizing layer {i+1}/{len(layers)}..')
        print('+------------------+--------------+------------+-----------+-------+')
        print('|       name       | weight_error | fp_inp_SNR | q_inp_SNR | time  |')
        print('+==================+==============+============+===========+=======+')

        layer = layers[i].to(dev)
        full = find_layers(layer,[torch.nn.Linear, lora.MergedLinear, lora.Linear])
        if args.true_sequential:
            sequential = [['self_attn.k_proj', 'self_attn.v_proj', 'self_attn.q_proj'], ['self_attn.o_proj'], ['mlp.up_proj', 'mlp.gate_proj'], ['mlp.down_proj']]
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
            for j in range(args.nsamples):
                #layer.attn.Wqkv.nbits = 16
                #layer.attn.out_proj.nbits = 16
                outs[j] = layer(inps[j].unsqueeze(0), **input_args)[0]
            for h in handles:
                h.remove()

            for name in subset:
                scale, zero, g_idx, error = gptq[name].fasterquant(percdamp=args.percdamp, groupsize=args.groupsize, actorder=args.act_order, name=name)
                quantizers['%s.%d.%s' % (prefix, i, name)] = (gptq[name].quantizer.cpu(), scale.cpu(), zero.cpu(), g_idx.cpu(), args.wbits, args.groupsize)

                if args.observe:
                    observer.submit(name=name, layerid=i, gptq=gptq[name], error=error)
                else:
                    gptq[name].free()

        # [ TODO ]
        # I am supposing layer's weight should be quantized and modified, we are statisting the error 
        # accumulated from the previous layers and compensate next layer
        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), **input_args
            )[0]

        layers[i] = layer.cpu()
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

                scale, zero, g_idx, error = gptq.fasterquant(percdamp=args.percdamp, groupsize=groupsize, actorder=args.act_order, name=name)

                table.add_row([wbits, groupsize, error])
                quantizers['%s.%d.%s' % (prefix, layerid, name)] = (gptq.quantizer.cpu(), scale.cpu(), zero.cpu(), g_idx.cpu(), wbits, groupsize)

            print(table.draw())
            print('\n')
            gptq.layer.to('cpu')
            gptq.free()

    model.config.use_cache = use_cache
    model=model.cuda()
    return quantizers


#@torch.no_grad()
def mpt_eval(model, argv_user, dev):
    print('Evaluating ...')
    sys.argv = argv_user
    import examples_ads
    if 'llama' in str(type(model)).lower():
        from examples_ads import run_llama_prompt as prompt_script
    else:
        from examples_ads import run_mpt_prompt as prompt_script
    print('\n\n\nCalling custom model with args ... \n', sys.argv)
    prompt_script.main(quant_model = model)


# TODO: perform packing on GPU
def mpt_pack(model, quantizers, wbits, groupsize):
    layers = find_layers(model, [torch.nn.Linear, lora.MergedLinear, lora.Linear])
    layers = {n: layers[n] for n in quantizers}
    quant.make_quant_linear(model, quantizers, wbits, groupsize)
    qlayers = find_layers(model, [quant.QuantLinear])
    print('Packing ...')
    for name in qlayers:
        print(name)
        quantizers[name], scale, zero, g_idx, _, _ = quantizers[name]
        # rewrite weight as quantized
        qlayers[name].oweight = qlayers[name].weight_qdq(layers[name], scale, zero, g_idx).cuda()
        layers[name].weight.data = qlayers[name].oweight
        assert (qlayers[name].oweight == qlayers[name].weight_qdq(layers[name], scale, zero, g_idx).cuda()).all()
        layers[name].nbits = qlayers[name].bits

        if DO_QDQ:
            if type(layers[name]) not in [nn.Linear]: #lora
                layers[name].scales = scale.T.contiguous()
                layers[name].qzeros = zero.T.contiguous()
                layers[name].g_idx = g_idx
        else:
            qlayers[name].pack_gpu(layers[name], scale, zero, g_idx)
            layers[name].scales = qlayers[name].scales
            layers[name].qzeros = qlayers[name].qzeros
            layers[name].g_idx = qlayers[name].g_idx
            layers[name].qweight = qlayers[name].qweight
            qlayers[name].oweight = None  # free memory
        
    del qlayers


    #quant.make_linear_qdq_back(model,layers)
    #if not DO_QDQ:
    #    quant.autotune_warmup_linear(model, transpose=False)

    print('Done.')
    return model.cuda()


def export_onnx(model, onnx_path, sample_inputs:tuple):
    torch.cuda.empty_cache()
    from pathlib import Path
    import shutil
    onnx_path = Path(onnx_path).absolute()
    assert onnx_path.suffix == '.onnx'
    inputs = {'input_ids':sample_inputs[0].to(model.device),"attention_mask":sample_inputs[1].to(model.device)}
    onnx_filepath = (onnx_path.parent/'tmp'/'tmp.onnx')
    onnx_filepath.parent.exists() and shutil.rmtree(onnx_filepath.parent)
    os.makedirs(onnx_filepath.parent)

    input_ids=inputs['input_ids']
    attention_mask=inputs['attention_mask']
    past_key_values=None
    onnx_inputs = (input_ids,past_key_values,attention_mask,None,None,None,True,False,False,False)
    if 'llama' in str(type(model)).lower():
        onnx_inputs = (input_ids,attention_mask,None,  None,None,None,False,False,  False,True)

    onnx_inp_names = ("input_ids", "attention_mask")
    onnx_out_names = ("logits",)
    onnx_dynamic_axes = {"input_ids": {0 : 'batch_size', 1:"seq_len"}, "attention_mask": {0 : 'batch_size', 1:"seq_len"}}
    torch.onnx.export(model=model, args=onnx_inputs, f=str(onnx_filepath), verbose=False, opset_version=16, input_names=onnx_inp_names, output_names=onnx_out_names, dynamic_axes=onnx_dynamic_axes)
    import onnx
    onnx_model=onnx.load(str(onnx_filepath))
    onnx_filepath = onnx_path
    onnx.save_model(onnx_model, str(onnx_filepath), save_as_external_data=True, all_tensors_to_one_file=True, location="graph.data", size_threshold=1024, convert_attribute=False)
    print(f"Exported onnx model to {onnx_filepath}")
    

def append_default_args():
    if '--wbits' not in sys.argv:
        sys.argv += ['--wbits', '4']
    
    if '--groupsize' not in sys.argv:
        sys.argv += ['--groupsize', '128']
    
    if '--nsamples' not in sys.argv:
        sys.argv += ['--nsamples', '512']

    #if '--export_onnx' not in sys.argv:
    #    sys.argv += ['--export_onnx', './mpt_onnx_q4/mpt.onnx']
 
    #if '--eval' not in sys.argv:
    #    sys.argv += ['--eval']

    #if '--save' not in sys.argv:
    #    sys.argv += ['--save', './mpt_q4']
    #if '--load' not in sys.argv:
    #    sys.argv += ['--load', './mpt_q4']

def process_forward_args(args):
    argv_user = args.forward_args
    import re
    key_with_space = re.findall(r'(".*"|\'.*\')', argv_user)
    argv_map = {}
    for idx, v in enumerate(key_with_space):
        argv_user = re.sub(v, f'____{idx}___', argv_user)
        argv_map[f'____{idx}___'] = v.strip('"')
    argv_user = argv_user.split(' ')
    argv_user = list(filter(None, argv_user))
    idx = 0
    for i in range(len(argv_user)):
        if argv_user[i] == f'____{idx}___':
            argv_user[i] = argv_map[f'____{idx}___']
            idx += 1
    args.forward_args = argv_user
    

if __name__ == '__main__':
    #,'--observe','--act-order'
    append_default_args()
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='mosaicml/mpt-7b', help='mpt model to load')
    parser.add_argument('--dataset', type=str, default='c4',choices=['wikitext2', 'ptb', 'c4'], help='Where to extract calibration data from.')
    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
    parser.add_argument('--nsamples', type=int, default=128, help='Number of calibration data samples.')
    parser.add_argument('--percdamp', type=float, default=.01, help='Percent of the average Hessian diagonal to use for dampening.')
    parser.add_argument('--nearest', action='store_true', help='Whether to run the RTN baseline.')
    parser.add_argument('--wbits', type=int, default=16, choices=[2, 3, 4, 8, 16], help='#bits to use for quantization; use 16 for evaluating base model.')
    parser.add_argument('--trits', action='store_true', help='Whether to use trits for quantization.')
    parser.add_argument('--groupsize', type=int, default=-1, help='Groupsize to use for quantization; default uses full row.')
    parser.add_argument('--eval', action='store_true', help='evaluate quantized model.')
    parser.add_argument('--save', type=str, default='', help='Save quantized checkpoint under this name.')
    parser.add_argument('--save_safetensors', type=str, default='', help='Save quantized `.safetensors` checkpoint under this name.')
    parser.add_argument('--load', type=str, default='', help='Load quantized model.')
    parser.add_argument('--check', action='store_true', help='Whether to compute perplexity during benchmarking for verification.')
    parser.add_argument('--sym', action='store_true', help='Whether to perform symmetric quantization.')
    parser.add_argument('--act-order', action='store_true', help='Whether to apply the activation order GPTQ heuristic')
    parser.add_argument('--true-sequential', action='store_true', help='Whether to run in true sequential model.')
    parser.add_argument('--new-eval', action='store_true', help='Whether to use the new PTB and C4 eval')
    parser.add_argument('--layers-dist', type=str, default='', help='Distribution of layers across GPUs. e.g. 2:1:1 for 2 layers on GPU 0, 1 layer on GPU 1, and 1 layer on GPU 2. Any remaining layers will be assigned to your last GPU.')
    parser.add_argument('--observe',
                        action='store_true',
                        help='Auto upgrade layer precision to higher precision, for example int2 to int4, groupsize 128 to 64. \
            When this feature enabled, `--save` or `--save_safetensors` would be disable.')
    parser.add_argument('--quant-directory', type=str, default=None, help='Specify the directory for export quantization parameters to toml format. `None` means no export by default.')
    parser.add_argument('--export_onnx', type=str, default=None, help='where does the onnx model save to.')
    parser.add_argument('--forward_args', type=str, default=None, help='args for run_prompts_mpt.py')

    args = parser.parse_args()
    process_forward_args(args)
    
    if args.layers_dist:
        gpu_dist = [int(x) for x in args.layers_dist.split(':')]
    else:
        gpu_dist = []

    if type(args.load) is not str:
        args.load = args.load.as_posix()

    if args.load:
        model,dataloader = load_quant(args.load, args.forward_args, args)
        model.eval()
    else:
        model,dataloader = get_mpt(args.model, args.forward_args, args.nsamples)
        model.eval()

    if not args.load and args.wbits < 16 and not args.nearest:
        tick = time.time()
        quantizers = mpt_sequential(model, dataloader, DEV)
        model = mpt_pack(model, quantizers, args.wbits, args.groupsize)
        del quantizers # free memory
        print(time.time() - tick)

    if args.eval:
        mpt_eval(model, args.forward_args, DEV)

    if args.quant_directory is not None:
        export_quant_table(quantizers, args.quant_directory)

    if not args.observe and args.save:
        model.save_pretrained(args.save)

    if args.export_onnx:
        export_onnx(model, args.export_onnx, dataloader[0])

    if not args.observe and args.save_safetensors:
        from safetensors.torch import save_file as safe_save
        state_dict = model.state_dict()
        state_dict = {k: v.clone().contiguous() for k, v in state_dict.items()}
        safe_save(state_dict, args.save_safetensors)

