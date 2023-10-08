from ..quant import make_mixbits_quant_linear, QuantLinear
from ..utils import find_layers, DEV, export_quant_table, gen_conditions
from ..quantization.gptq import GPTQ, Observer
from texttable import Texttable
import os
if "CUDA_VISIBLE_DEVICES" not in os.environ:  # NOQA
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # NOQA

import torch.nn as nn
import torch
import numpy as np

import argparse
import time
from pathlib import Path
import json

import sys
# sys.path.append(os.path.dirname(__file__)) # NOQA
print("adding path", os.getcwd())
sys.path.append(os.getcwd())   # NOQA
import loralib as lora # NOQA


NEED_CHECK_PACK = False


def get_mpt(model, argv_user, nsamples):
    if 'ckpt/mpt-' not in argv_user[argv_user.index('--model_name_or_path')+1]:
        lora_ind = argv_user.index('--use_lora')
        argv_user[lora_ind+1] = 'False'

    import examples_ads
    from examples_ads import run_mpt_prompt
    argv_user.insert(0, run_mpt_prompt.__file__)
    argv_back = sys.argv
    sys.argv = argv_user

    os.environ['init_device'] = "cpu"
    model, data_sets = run_mpt_prompt.main(True)
    new_data = []
    for idx, indata in enumerate(data_sets):
        if idx >= nsamples:
            break
        input_ = (torch.tensor([indata["input_ids"]]), torch.tensor([indata["attention_mask"]]))
        new_data.append(input_)
    return model.half(), new_data


def load_quant(qmodel, argv_user, args):
    argv_user[argv_user.index('--model_name_or_path')+1] = os.path.abspath(qmodel)
    # do not init model twice as it slow initialization
    import torch.nn.init
    torch.nn.init.kaiming_uniform_ = lambda x, *args, **kwargs: x
    torch.nn.init.uniform_ = lambda x, *args, **kwargs: x
    torch.nn.init.kaiming_normal_ = lambda x, *args, **kwargs: x
    model, dataloader = get_mpt(args.model, argv_user, args.nsamples)
    import transformers
    layers = find_layers(model, layers=[torch.nn.Linear, lora.MergedLinear, lora.Linear])
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


@torch.no_grad()
def mpt_sequential(model, dataloader, mix_qlayer_conf, dev):
    print('Starting ...')
    model = model.cpu()
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.get_decoder().blocks

    model.get_decoder().wte = model.get_decoder().wte.to(dev)
    model.get_decoder().norm_f = model.get_decoder().norm_f.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    # model.config.max_seq_len
    inps = torch.zeros((args.nsamples, dataloader[0][0].shape[-1], model.config.d_model), dtype=dtype)
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):

        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp.cpu()
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            # cache['position_ids'] = kwargs['position_ids']
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
    # position_ids = cache['position_ids']

    print('Ready.')

    quantizers = {}
    observer = Observer()
    for i in range(len(layers)):
        if str(i+1) in mix_qlayer_conf:
            layer_key = str(i+1)
            args.wbits = mix_qlayer_conf[layer_key].get('wbits', args.wbits)
            args.groupsize = mix_qlayer_conf[layer_key].get('groupsize', args.groupsize)

        print(f'Quantizing layer {i+1}/{len(layers)}..')
        print('+------------------+--------------+------------+-----------+-------+')
        print('|       name       | weight_error | fp_inp_SNR | q_inp_SNR | time  |')
        print('+==================+==============+============+===========+=======+')

        layer = layers[i].to(dev)
        full = find_layers(layer, [torch.nn.Linear, lora.MergedLinear, lora.Linear])
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
            for j in range(args.nsamples):
                layer.attn.Wqkv.nbits = 16
                layer.attn.out_proj.nbits = 16
                outs[j] = layer(inps[j].unsqueeze(0).to(dev), attention_mask=attention_mask,  # position_ids=position_ids
                                )[0].cpu()
            for h in handles:
                h.remove()

            for name in subset:
                scale, zero, g_idx, error = gptq[name].fasterquant(
                    percdamp=args.percdamp, groupsize=args.groupsize, actorder=args.act_order, name=name)
                quantizers['transformer.blocks.%d.%s' % (i, name)] = (
                    gptq[name].quantizer.cpu(), scale.cpu(), zero.cpu(), g_idx.cpu(), args.wbits, args.groupsize)

                if args.observe:
                    observer.submit(name=name, layerid=i, gptq=gptq[name], error=error)
                else:
                    gptq[name].free()

        # [ TODO ]
        # I am supposing layer's weight should be quantized and modified, we are statisting the error
        # accumulated from the previous layers and compensate next layer
        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0).to(dev), attention_mask=attention_mask,  # position_ids=position_ids
                            )[0].cpu()

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

                scale, zero, g_idx, error = gptq.fasterquant(
                    percdamp=args.percdamp, groupsize=groupsize, actorder=args.act_order, name=name)

                table.add_row([wbits, groupsize, error])
                quantizers['transformer.blocks.%d.%s' % (layerid, name)] = (
                    gptq.quantizer.cpu(), scale.cpu(), zero.cpu(), g_idx.cpu(), wbits, groupsize)

            print(table.draw())
            print('\n')
            gptq.layer.to('cpu')
            gptq.free()

    model.config.use_cache = use_cache
    return quantizers


# @torch.no_grad()
def mpt_eval(model, argv_user, dev):
    print('Evaluating ...')
    sys.argv = argv_user
    import examples_ads
    from examples_ads import run_mpt_prompt
    run_mpt_prompt.main(quant_model=model.to(dev))


# TODO: perform packing on GPU
def mpt_pack(model, quantizers):
    layers = find_layers(model, [torch.nn.Linear, lora.MergedLinear, lora.Linear])
    layers = {n: layers[n] for n in quantizers}
    quant_info = {key: {"wbits": value[-2], "groupsize": value[-1]} for key, value in quantizers.items()}
    make_mixbits_quant_linear(model, quantizers, quant_info)
    qlayers = find_layers(model, [QuantLinear])
    print('Packing ...')
    for name in qlayers:
        print(name)
        quantizers[name], scale, zero, g_idx, _, _ = quantizers[name]
        # rewrite weight as quantized
        if NEED_CHECK_PACK:
            qlayers[name].oweight = qlayers[name].weight_qdq(layers[name], scale, zero, g_idx).cuda()
            layers[name].weight.data = qlayers[name].oweight
            assert (qlayers[name].oweight == qlayers[name].weight_qdq(layers[name], scale, zero, g_idx).cuda()).all()
            layers[name].nbits = qlayers[name].bits

        qlayers[name].pack_gpu(layers[name], scale, zero, g_idx)

    # quant.make_linear_qdq_back(model,layers)
    # quant.autotune_warmup_linear(model, transpose=False)

    print('Done.')
    return model, quant_info


def export_onnx(model, onnx_path, sample_inputs: tuple):
    # model = model.cpu().float()
    model = model.cuda()
    from pathlib import Path
    import shutil
    onnx_filepath = Path(onnx_path).absolute()
    assert onnx_filepath.suffix == '.onnx'
    inputs = {'input_ids': sample_inputs[0].to(model.device), "attention_mask": sample_inputs[1].to(model.device)}
    onnx_filepath_export_multi_files_tmp = onnx_filepath.parent/'tmp/tmp.onnx'
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

    onnx_filepath.exists() and onnx_filepath.unlink()
    (onnx_filepath.parent/'mpt_ext.data').exists() and (onnx_filepath.parent/'mpt_ext.data').unlink()
    onnx.save_model(onnx_model, str(onnx_filepath), save_as_external_data=True, all_tensors_to_one_file=True,
                    location="mpt_ext.data", size_threshold=1024, convert_attribute=False)


def append_default_args():
    if '--wbits' not in sys.argv:
        sys.argv += ['--wbits', '4']

    if '--groupsize' not in sys.argv:
        sys.argv += ['--groupsize', '128']

    if '--nsamples' not in sys.argv:
        sys.argv += ['--nsamples', '512']

    # if '--export_onnx' not in sys.argv:
    #    sys.argv += ['--export_onnx', './mpt_onnx_q4/mpt.onnx']
 #
    # if '--eval' not in sys.argv:
    #    sys.argv += ['--eval']

    # if '--save' not in sys.argv:
    #    sys.argv += ['--save', './mpt_q4']
    # if '--load' not in sys.argv:
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
    # ,'--observe','--act-order'
    append_default_args()
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='mosaicml/mpt-7b', help='mpt model to load')
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
    parser.add_argument('--mix_qlayer_conf', type=str, default=None,
                        help='Mix quantization layer configuration.(groupsize,wbits)')
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
        model, dataloader = load_quant(args.load, args.forward_args, args)
        model.eval()
    else:
        model, dataloader = get_mpt(args.model, args.forward_args, args.nsamples)
        model.eval()

    if not args.load and args.wbits < 16 and not args.nearest:
        if args.mix_qlayer_conf:
            args.mix_qlayer_conf = json.load(open(args.mix_qlayer_conf))
        else:
            args.mix_qlayer_conf = {}
        tick = time.time()
        quantizers = mpt_sequential(model, dataloader, args.mix_qlayer_conf, DEV)
        model, quant_info = mpt_pack(model, quantizers)
        print(time.time() - tick)

    if args.quant_directory is not None:
        export_quant_table(quantizers, args.quant_directory)

    if not args.observe and args.save:
        model.save_pretrained(args.save)
        open(args.save+"/quant.op.json", 'w').write(json.dumps(quant_info))

    if args.eval:
        mpt_eval(model, args.forward_args, DEV)

    if args.export_onnx:
        export_onnx(model, args.export_onnx, dataloader[0])

    if not args.observe and args.save_safetensors:
        from safetensors.torch import save_file as safe_save
        state_dict = model.state_dict()
        state_dict = {k: v.clone().contiguous() for k, v in state_dict.items()}
        safe_save(state_dict, args.save_safetensors)
