import datetime
import gc
import os
import tqdm
import concurrent
import queue as theading_queue

import torch
import torch.cuda.streams
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from datasets import load_dataset
import copy
from pathlib import Path

from ...utils import comm_utils
from ...utils.logger import get_logger
logger = get_logger("qllm")

def set_seed(seed):
    import numpy
    import random
    torch.manual_seed(seed)
    random.seed(seed)
    numpy.random.seed(seed)

def wrap_tokenizer(input_args):
    tokenizer, x, ctx_size = input_args
    out= tokenizer(x, return_tensors='pt', truncation=True, padding=True, max_length=ctx_size)
    return out

def sample_rp1t(tokenizer, size=128, ctx_size=2048, save_path="./hessian_path/", nproc=1, seed=0):
    import re
    set_seed(seed)
    Path(save_path).mkdir(parents=True, exist_ok=True)
    dataset_name = 'togethercomputer/RedPajama-Data-1T-Sample'
    named_hash = f"{tokenizer.name_or_path}_{dataset_name}_{size}_{ctx_size}_{seed}"
    named_hash = re.sub(r"[^0-9a-zA-Z_-]", "", named_hash)
    cache_path = Path(save_path)/named_hash

    if cache_path.exists():
        return torch.load(cache_path, map_location="cpu", weights_only=True)
    dataset = load_dataset(dataset_name, split='train', trust_remote_code=True)
    devset = torch.zeros((size, ctx_size), dtype=torch.int64)
    saved = 0
    nproc = max(nproc, 2)
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=nproc)
    pbar = tqdm.tqdm(total=size, desc='sampling dataset...')
    while saved < size:
        seqs = [(tokenizer, dataset[torch.randint(len(dataset), (size,))]['text'], ctx_size) for _ in range(nproc)]
        tokens = executor.map(wrap_tokenizer, seqs)

        for token in (tokens):
            lens = token.attention_mask.sum(dim=-1)
            good = torch.where(lens == ctx_size)[0]
            if len(good) > 0:
                if saved + len(good) > size:
                    good = good[:size - saved]
                devset[saved:saved + len(good)] = token.input_ids[good]
                saved += len(good)
                pbar.update(len(good))
    torch.save(devset, cache_path)
    executor.shutdown(wait=True)
    return devset


def register_H_hook(module, device):
    n = module.in_features
    H = torch.zeros(n, n, dtype=torch.float64, device=device)
    mu = torch.zeros(n, dtype=torch.float64, device=device)
    ct = 0

    def H_hook(module, x):
        nonlocal H, mu, ct, n
        x = x[0].reshape(-1, n).to(torch.float64)
        mu.add_(x.sum(dim=0))
        H.addmm_(x.T, x)
        ct += len(x)

    hook = module.register_forward_pre_hook(H_hook)

    def done():
        nonlocal H, mu, ct, hook
        hook.remove()
        return H.cpu(), mu.cpu(), ct

    return done


def flat_to_sym(V, N):
    A = torch.zeros(N, N, dtype=V.dtype, device=V.device)
    idxs = torch.tril_indices(N, N, device=V.device)
    A[idxs.unbind()] = V
    A[idxs[1, :], idxs[0, :]] = V
    return A


def sym_to_flat(A):
    N = A.shape[-1]
    idxs = torch.tril_indices(N, N, device=A.device)
    return A[idxs.unbind()]

@torch.inference_mode()
def forward_layer(layer, position_ids, attention_mask, layer_args, bs, device, in_q, out_q):
    torch.set_grad_enabled(False)
    layer = layer.to(device)
    position_ids = position_ids.to(device)
    attention_mask = attention_mask.to(device)
    for k,v in layer_args.items():
        if isinstance(v, torch.Tensor):
            layer_args[k] = v.to(device)
    if 'position_embeddings' in layer_args:
        layer_args['position_embeddings'] = tuple([
            i.to(device) for i in  layer_args['position_embeddings']])

    # register hooks
    done_qkv = register_H_hook(layer.self_attn.q_proj, device)
    done_o = register_H_hook(layer.self_attn.o_proj, device)
    done_up = register_H_hook(layer.mlp.up_proj, device)
    done_down = register_H_hook(layer.mlp.down_proj, device)

    while True:
        dev_emb = in_q.get()
        if dev_emb is None:
            layer = layer.cpu()
            position_ids = position_ids.cpu()
            attention_mask = attention_mask.cpu()
            out_q.put({'qkv': done_qkv(), 'o': done_o(), 'up': done_up(), 'down': done_down()})
            return

        assert len(dev_emb) % bs == 0
        for i in range(len(dev_emb) // bs):
            batch = dev_emb[i * bs:(i + 1) * bs].to(device)
            with torch.cuda.stream(torch.cuda.Stream(device=device)):
                layer_args['position_ids'] = position_ids
                layer_args['attention_mask'] = attention_mask
                layer_args['use_cache'] = False
                layer_args['output_attentions'] = False
                output = layer(batch, **layer_args)[0]
                dev_emb[i:i + bs] = output.cpu()
                del output

            # clear cache every 4 batches
            if i % (bs * 4) == 0:
                torch.cuda.empty_cache()


def accumulate(in_q, ngpus, args, transformer_layer_index):
    Hs = {}
    mus = {}
    cts = {}

    for i in range(ngpus):
        out = in_q.get()
        if i == 0:
            for key in out:
                Hs[key] = torch.zeros(out[key][0].shape, dtype=out[key][0].dtype)
                mus[key] = torch.zeros(out[key][1].shape, dtype=out[key][1].dtype)
                cts[key] = 0
        for key in out:
            Hs[key].add_(out[key][0])
            mus[key].add_(out[key][1])
            cts[key] += out[key][2]

    # keys = list(Hs.keys())

    for key in Hs:
        mus[key].div_(cts[key])
        Hs[key].div_(cts[key])
        Hs[key].addmm_(-mus[key].unsqueeze(-1), mus[key].unsqueeze(0))
        save_path = f"{args.scratch_path}/{transformer_layer_index}_{key}.pt" \
            if args.scratch_path is not None else f"{args.save_path}/{transformer_layer_index}_{key}.pt"
        torch.save({
            'flatH': sym_to_flat(Hs[key].to(torch.float32)),
            'mu': mus[key].to(torch.float32),
            'n': Hs[key].shape[0],
            'ct': cts[key]
        }, save_path)

    del Hs, mus, cts, out


def process_collect_hessian(args, embed_forward_func, model, tokenizer, dev):
    from .merge_hessian import main as merge_hessian
    devset_size = args.devset_size
    save_dir = args.save_path
    tokenizer.pad_token = tokenizer.eos_token
    devset = sample_rp1t(tokenizer, args.devset_size, args.ctx_size, args.save_path, nproc=args.sample_proc, seed=0)

    for idx, start in enumerate(range(0, devset_size, args.iter_size)):
        logger.info(f"Processing group {idx} with start {start}")
        args.devset_size = min(args.iter_size, devset_size + start)
        args.save_path = save_dir + f'_{idx}'
        args.seed = idx
        out = partion_collect_hessian(args, embed_forward_func, model, devset[start:start+args.devset_size], dev)

    if devset_size > args.iter_size:
        args.base_dir = save_dir+'_'
        args.save_dir = save_dir
        args.groups = list(range(idx+1))
        logger.info(f"Merging hessian {args.base_dir} {args.groups} {args.save_dir}")
        merge_hessian(args)
    
    args.save_path = save_dir
    from .inv_hessian import main as inv_hessian
    args.load_hessian_dir = save_dir
    args.store_inv_hessian_dir = save_dir + '_inv'
    args.enable_perm = True
    inv_hessian(args)
    return out

def partion_collect_hessian(args, embed_forward_func, model, devset, dev):
    save_path = Path(args.save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    if (save_path/"done.txt").exists():
        dev_emb, attention_layers, layer_args = embed_forward_func(model, [(devset[:1],)])
        return attention_layers, layer_args
    dev_emb, attention_layers, layer_args = embed_forward_func(model, [(devset,)])
    comm_utils.clear_memory((devset,))

    free_gpu_memory, total_gpu_memory = torch.cuda.mem_get_info()
    logger.info(f"free_gpu_memory: {free_gpu_memory/1024**3:.2f}GB, " +
                f"total_gpu_memory: {total_gpu_memory/1024**3:.2f}GB")

    logger.info(f"dev_emb dtype: {dev_emb.dtype}")
    dev_emb.share_memory_()

    position_ids = torch.arange(args.ctx_size, dtype=torch.int64)[None, :] + \
        torch.zeros(args.batch_size, args.ctx_size, dtype=torch.int64)

    
    kargs = (None, (args.batch_size, args.ctx_size), dev_emb[0:args.batch_size], 0)
    kwargs = {}
    if hasattr(model.config, 'sliding_window'):
        kwargs = dict(sliding_window=model.config.sliding_window)
    attention_mask = _prepare_4d_causal_attention_mask(*kargs, **kwargs)
    for transformer_layer_index in tqdm.trange(len(attention_layers), 
            desc='hessian collection--processing layers'):
        transformer_layer = attention_layers[transformer_layer_index]
        # check that there are four layers, as expected
        # assert (len([m for m in transformer_layer.modules() if isinstance(m, torch.nn.Linear)]) == 7)

        chunk_size = min(args.chunk_size, len(dev_emb))
        ngpus = min(torch.cuda.device_count(), len(dev_emb) // chunk_size)

        executor = concurrent.futures.ThreadPoolExecutor(max_workers=(ngpus+2))
        in_q = theading_queue.Queue()
        out_q = theading_queue.Queue()

        accumulate_proc = executor.submit(accumulate, out_q, ngpus, args, transformer_layer_index)

        forward_procs = []
        for i in range(ngpus):
            p = executor.submit(
                forward_layer,
                copy.deepcopy(transformer_layer), 
                copy.deepcopy(position_ids), 
                copy.deepcopy(attention_mask), 
                copy.deepcopy(layer_args), 
                args.batch_size, i, in_q, out_q)
            forward_procs.append(p)

        assert len(dev_emb) % args.batch_size == 0 and chunk_size % args.batch_size == 0
        i = 0
        while i < len(dev_emb):
            next = min(i + chunk_size, len(dev_emb))
            in_q.put(dev_emb[i:next])
            i = next

        for _ in range(ngpus):
            in_q.put(None)

        for p in forward_procs:
            p.result()

        transformer_layer.cpu()
        accumulate_proc.result()

        gc.collect()
        torch.cuda.empty_cache()

        if args.save_activations and (
            transformer_layer_index % args.act_save_rate == 0 or transformer_layer_index == len(attention_layers) - 1
        ):
            if args.scratch_path is not None:
                if os.path.exists(f'{args.scratch_path}/dev_activations.pt'):
                    print('not saving layer since disk is too slow')
                else:
                    torch.save({
                        'dev_emb': dev_emb,
                        'after_layer': transformer_layer_index,
                        'timestamp': str(datetime.datetime.now())
                    }, f'{args.scratch_path}/dev_activations.pt')
                    move_q.put((f'{args.scratch_path}/dev_activations.pt', f'{args.save_path}/dev_activations.pt'))
            else:
                torch.save({
                    'dev_emb': dev_emb,
                    'after_layer': transformer_layer_index,
                    'timestamp': str(datetime.datetime.now())
                }, f'{args.save_path}/dev_activations.pt')
    executor.shutdown(wait=True)
    # save done.txt to indicate that the process is done
    with open(save_path/"done.txt", "w") as f:
        f.write("done")
    return attention_layers, layer_args