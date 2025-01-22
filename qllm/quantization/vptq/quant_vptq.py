from pathlib import Path
import torch
import tqdm
import concurrent
import queue as theading_queue

from ...utils import comm_utils
from ..quant_frame_base import QuantFrameBase
from ._vptq_quantizer import InternalVPTQQuantizer
from ...utils.logger import get_logger
from ...utils import find_layers

logger = get_logger('qllm')


class VPTQQuant(QuantFrameBase):

    def __init__(self, config) -> None:
        super().__init__()
        self.quant_config = config

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def hijack_internal_block(self, vptq, subset, layer_block, inps, layer_kwargs):
        dev = next(layer_block.parameters()).device

        def add_batch(name):
            def tmp(_, inp, out):
                vptq[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in subset:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(len(inps)):
            _ = layer_block(inps[j].unsqueeze(0).to(dev), **layer_kwargs)
        for h in handles:
            h.remove()
        
    def collect_hessian_pre(self, model, model_prefix, dev):
        if self.quant_config.hessian_path is not None and self.quant_config.inv_hessian_path is not None:
            logger.info("read cached Hessian data")
            _, attention_layers, layer_input_args = self.hijack_block_inputs(model, [(torch.tensor((1, 1), dtype=torch.int64), )], model_prefix, "cpu")
            return attention_layers, layer_input_args
        
        # sample_args = f"--batch_size 2 --devset_size 3072 --ctx_size 8192 \
        #     --base_model {self.quant_config.model_name} --save_path Meta-Llama-3-70B-1 \
        #         --act_save_rate 50 --sample_proc 4"
        from .qllm_hessian import process_collect_hessian
        sample_args = self.quant_config.hessian_config
        # SampleArgs = type("ARGS", (object,), {"init1": lambda: None})
        # sample_args = SampleArgs()
        # sample_args.batch_size = 2
        # sample_args.devset_size = 32#3072
        # sample_args.iter_size = 16
        # sample_args.ctx_size = 8192
        # sample_args.chunk_size = 256
        sample_args.base_model = self.quant_config.model_name
        # sample_args.act_save_rate = 50
        # sample_args.sample_proc = 4
        # sample_args.scratch_path = None
        # sample_args.save_activations = None
        sample_args.save_path = f"./hessian_path/{sample_args.base_model}_{sample_args.devset_size}_{sample_args.ctx_size}"
        
        self.quant_config.hessian_path = sample_args.save_path
        self.quant_config.inv_hessian_path = sample_args.save_path+"_inv"

        def embed_func(model, dataloader):
            return self.hijack_block_inputs(model, dataloader, model_prefix, "cpu")

        if (Path(sample_args.save_path)/"done.txt").exists():
            logger.info("Hessian already collected")
            _, attention_layers, layer_input_args = self.hijack_block_inputs(model, [(torch.tensor((1, 1), dtype=torch.int64), )], model_prefix, "cpu")
            return attention_layers, layer_input_args

        logger.info("Collecting Hessian====ctx_size=%s, devset_size=%s", sample_args.ctx_size, sample_args.devset_size)
        output = process_collect_hessian(sample_args, embed_func, model, self.tokenizer, dev)
        with open(Path(sample_args.save_path)/"done.txt", "w") as f:
            f.write("done")
        comm_utils.clear_memory()
        free_gpu_memory, total_gpu_memory = torch.cuda.mem_get_info()
        logger.info(f"free_gpu_memory: {free_gpu_memory/1024**3:.2f}GB, " +
                f"total_gpu_memory: {total_gpu_memory/1024**3:.2f}GB")        
        return output

    def parallel_quantize(self, quantize_layer, attention_layers, num_gpus, dev):
        # init multiprocessing
        processes = []
        # executor = concurrent.futures.ThreadPoolExecutor(max_workers=(num_gpus+2))
        import multiprocessing
        # layer init
        torch.inverse(torch.ones((1, 1), device=dev))
        executor = concurrent.futures.ProcessPoolExecutor(max_workers=(num_gpus), mp_context=multiprocessing.get_context("spawn"))

        pbar = tqdm.tqdm(total=len(attention_layers), desc=f"running VPTQ on {num_gpus} GPUs")
        output_queue = theading_queue.Queue()
        quant_tmp = Path("quant_tmp")
        quant_tmp.mkdir(exist_ok=True)
        for i in range(num_gpus):
            output_queue.put(i) # poison pill
        def fetch_next_task(future):
            comm_utils.clear_memory()
            pbar.update(1)
            output_queue.put(future.gpu_idx)
            torch.save(future.result(), quant_tmp/f"layer_{future.layer_idx}.pt")

        for layer_idx,layer in enumerate(attention_layers):
            if (quant_tmp/f"layer_{layer_idx}.pt").exists():
                attention_layers[layer_idx] = torch.load(quant_tmp/f"layer_{layer_idx}.pt")
                pbar.update(1)
                continue
            free_gpu_id = output_queue.get()
            future_task = executor.submit(
                quantize_layer,
                (layer, layer_idx),
                self.quant_config,
                self.quant_config,
                dev=torch.device(f"cuda:{free_gpu_id}"),
            )
            future_task.gpu_idx = free_gpu_id
            future_task.layer_idx = layer_idx
            future_task.add_done_callback(fetch_next_task)
            processes.append(future_task)
        for p in processes:
            attention_layers[p.layer_idx] = p.result() # wait for the last task to finish
        pbar.close()
        executor.shutdown(wait=True)
        return {}

    @torch.inference_mode()
    def do_quantize(self, model, dataloader, model_prefix, dev):
        # lazy load
        try:
            from vptq.utils.pack import absorb_perm, pack_model
            from vptq import VQuantLinear
        except ImportError:
            logger.warning("VPTQ is not installed, skipping VPTQ quantization")
            return {}
        attention_layers, layer_input_args = self.collect_hessian_pre(model, model_prefix, dev)
        print('Ready.')

        # calculate task allocation
        total_layers = len(attention_layers)
        num_gpus = min(self.quant_config.num_gpus, total_layers, torch.cuda.device_count())

        vptq_quantizer = InternalVPTQQuantizer()
        quantize_layer = vptq_quantizer.quantize_layer
        quantizers = {}

        if num_gpus > 1:
            self.parallel_quantize(quantize_layer, attention_layers, num_gpus, dev)
        else:        
            
            for layer_idx in tqdm.trange((len(attention_layers)), desc="running VPTQ"):
                attention_layers[layer_idx] = quantize_layer(
                    (attention_layers[layer_idx], layer_idx), self.quant_config, self.quant_config, dev
                )

        config_for_layers = {k:v.init_args for k,v in  find_layers(model, [VQuantLinear]).items()}
        MetaConf = type(
            "MetaConf",
            (object,),
            {
                "version": "AUTO",
                "quant_method": "vptq",
                "bits": 2,
                "to_dict": lambda self: self.config_for_layers,
                "to_meta": property(lambda self: self),
            },
        )
        meta_conf = MetaConf()
        meta_conf.config_for_layers = {"config_for_layers": config_for_layers}
        self.quant_config = meta_conf
        model.quant_config_by_layer = {}
        model = pack_model(model, from_type=torch.uint16, to_type=torch.uint16, as_type=torch.int16)
        # if self.quant_config.absorb_perm:
        #     model = absorb_perm(model)
        return quantizers
