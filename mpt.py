import argparse
import time
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"]="3"

import torch
import torch.nn as nn
import quant

import sys
sys.path.append('/home/jicwen/GPTQ-for-LLaMa')
sys.path.append('/home/jicwen/AdsLR_MTL')

from gptq import GPTQ, Observer
from utils import find_layers, DEV, set_seed, get_wikitext2, get_ptb, get_c4, get_ptb_new, get_c4_new, get_loaders, export_quant_table, gen_conditions
from texttable import Texttable
DO_QDQ = False


def get_mpt(model, argv_user,nsamples):
    import examples_ads
    argv_back = sys.argv
    sys.argv = argv_user
    from examples_ads import run_mpt_prompt
    model,data_sets = run_mpt_prompt.main(True)
    new_data = []
    for idx, indata in enumerate(data_sets):
        if idx>=nsamples:break
        input_ = (torch.tensor([indata["input_ids"]]), torch.tensor([indata["attention_mask"]]))            
        new_data.append(input_)
    #model.config.max_seq_len = 128
    #for key in list(data.features.keys()):
    #    if key not in ['input_ids','attention_mask']:
    #        data=data.remove_columns(key)
    return model.half(),new_data
    def skip(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    import transformers
    from transformers import AutoModelForCausalLM
    config = transformers.AutoConfig.from_pretrained(model,trust_remote_code=True)
    #config.update({"max_seq_len": 4096})
    model = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch.float16,config=config,trust_remote_code=True)

    return model

def load_quant(model, argv_user, args):
    if argv_user[argv_user.index('--model_name_or_path')+1] not in ['ckpt/mpt-7b/', 'ckpt/mpt-7b']:
        os.environ["nbits"] = "4"

    model,dataloader = get_mpt(args.model, argv_user, args.nsamples)
    import loralib as lora
    layers = find_layers(model,layers=[lora.GptqQuantLinear])
    #quant.replace_quant_linear_layer(model,layers ,args.wbits, args.groupsize)
    #quant.autotune_warmup_linear(model, transpose=False)
    return model,dataloader

@torch.no_grad()
def mpt_sequential(model, dataloader, dev):
    print('Starting ...')
    model=model.cpu()
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.get_decoder().blocks

    model.get_decoder().wte = model.get_decoder().wte.to(dev)
    model.get_decoder().norm_f = model.get_decoder().norm_f.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    #model.config.max_seq_len
    inps = torch.zeros((args.nsamples, dataloader[0][0].shape[-1], model.config.d_model), dtype=dtype, device=dev)
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):

        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            #cache['position_ids'] = kwargs['position_ids']
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.get_decoder().wte = model.get_decoder().wte.cpu()
    model.get_decoder().norm_f = model.get_decoder().norm_f.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    #position_ids = cache['position_ids']

    print('Ready.')

    quantizers = {}
    observer = Observer()
    for i in range(len(layers)):

        print(f'Quantizing layer {i+1}/{len(layers)}..')
        print('+------------------+--------------+------------+-----------+-------+')
        print('|       name       | weight_error | fp_inp_SNR | q_inp_SNR | time  |')
        print('+==================+==============+============+===========+=======+')

        layer = layers[i].to(dev)
        full = find_layers(layer)
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
                layer.attn.Wqkv.nbits = 16
                layer.attn.out_proj.nbits = 16
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, #position_ids=position_ids
                )[0]
            for h in handles:
                h.remove()

            for name in subset:
                scale, zero, g_idx, error = gptq[name].fasterquant(percdamp=args.percdamp, groupsize=args.groupsize, actorder=args.act_order, name=name)
                quantizers['transformer.blocks.%d.%s' % (i, name)] = (gptq[name].quantizer.cpu(), scale.cpu(), zero.cpu(), g_idx.cpu(), args.wbits, args.groupsize)

                if args.observe:
                    observer.submit(name=name, layerid=i, gptq=gptq[name], error=error)
                else:
                    gptq[name].free()

        # [ TODO ]
        # I am supposing layer's weight should be quantized and modified, we are statisting the error 
        # accumulated from the previous layers and compensate next layer
        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, #position_ids=position_ids
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
                quantizers['transformer.blocks.%d.%s' % (layerid, name)] = (gptq.quantizer.cpu(), scale.cpu(), zero.cpu(), g_idx.cpu(), wbits, groupsize)

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
    from examples_ads import run_mpt_prompt
    run_mpt_prompt.main(quant_model = model)


# TODO: perform packing on GPU
def mpt_pack(model, quantizers, wbits, groupsize):
    layers = find_layers(model)
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


    #quant.make_linear_qdq_back(model,layers)
    #if not DO_QDQ:
    #    quant.autotune_warmup_linear(model, transpose=False)

    print('Done.')
    return model.cuda()


def benchmark(model, input_ids, check=False):
    input_ids = input_ids.to(model.gpus[0] if hasattr(model, 'gpus') else DEV)
    torch.cuda.synchronize()

    cache = {'past': None}

    def clear_past(i):

        def tmp(layer, inp, out):
            if cache['past']:
                cache['past'][i] = None

        return tmp

    for i, layer in enumerate(model.transformer.blocks):
        layer.register_forward_hook(clear_past(i))

    print('Benchmarking ...')

    if check:
        loss = nn.CrossEntropyLoss()
        tot = 0.

    def sync():
        if hasattr(model, 'gpus'):
            for gpu in model.gpus:
                torch.cuda.synchronize(gpu)
        else:
            torch.cuda.synchronize()

    max_memory = 0
    with torch.no_grad():
        attention_mask = torch.ones((1, input_ids.numel()), device=DEV)
        times = []
        for i in range(input_ids.numel()):
            tick = time.time()
            out = model(input_ids[:, i:i + 1], past_key_values=cache['past'], attention_mask=attention_mask[:, :(i + 1)].reshape((1, -1)))
            sync()
            times.append(time.time() - tick)
            print(i, times[-1])
            if hasattr(model, 'gpus'):
                mem_allocated = sum(torch.cuda.memory_allocated(gpu) for gpu in model.gpus) / 1024 / 1024
            else:
                mem_allocated = torch.cuda.memory_allocated() / 1024 / 1024
            max_memory = max(max_memory, mem_allocated)
            if check and i != input_ids.numel() - 1:
                tot += loss(out.logits[0].to(DEV), input_ids[:, (i + 1)].to(DEV)).float()
            cache['past'] = list(out.past_key_values)
            del out
        sync()
        print('Median:', np.median(times))
        if check:
            print('PPL:', torch.exp(tot / (input_ids.numel() - 1)).item())
            print('max memory(MiB):', max_memory)

argv_user = sys.argv
if argv_user[argv_user.index('--model_name_or_path')+1] not in ['ckpt/mpt-7b/', 'ckpt/mpt-7b']:
    lora_ind = argv_user.index('--use_lora')
    argv_user[lora_ind+1] = 'False'

if __name__ == '__main__':

    #,'--observe','--act-order'
    sys.argv = ['', '--wbits', '4', '--groupsize', '128','--eval', '--nsamples','512','--load', './']
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
    parser.add_argument('--benchmark', type=int, default=0, help='Number of tokens to use for benchmarking.')
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

    args = parser.parse_args()

    if args.layers_dist:
        gpu_dist = [int(x) for x in args.layers_dist.split(':')]
    else:
        gpu_dist = []

    if type(args.load) is not str:
        args.load = args.load.as_posix()

    if args.load:
        model,dataloader = load_quant(args.model, argv_user, args)
        model.eval()
    else:
        model,dataloader = get_mpt(args.model, argv_user, args.nsamples)
        model.eval()

    #dataloader, testloader = get_loaders(args.dataset, nsamples=args.nsamples, seed=args.seed, model=args.model, seqlen=model.config.max_seq_len)

    if not args.load and args.wbits < 16 and not args.nearest:
        tick = time.time()
        quantizers = mpt_sequential(model, dataloader, DEV)
        model = mpt_pack(model, quantizers, args.wbits, args.groupsize)
        print(time.time() - tick)

    if args.benchmark:
        gpus = [torch.device('cuda:%d' % i) for i in range(torch.cuda.device_count())]
        model = model.to(DEV)
        if args.benchmark:
            input_ids = next(iter(dataloader))[0][:, :args.benchmark]
            benchmark(model, input_ids, check=args.check)

    if args.eval:
        mpt_eval(model, argv_user, DEV)

    if args.quant_directory is not None:
        export_quant_table(quantizers, args.quant_directory)

    if not args.observe and args.save:
        mpt_pack(model, quantizers, args.wbits, args.groupsize)
        torch.save(model.state_dict(), args.save)

    if not args.observe and args.save_safetensors:
        mpt_pack(model, quantizers, args.wbits, args.groupsize)
        from safetensors.torch import save_file as safe_save
        state_dict = model.state_dict()
        state_dict = {k: v.clone().contiguous() for k, v in state_dict.items()}
        safe_save(state_dict, args.save_safetensors)

