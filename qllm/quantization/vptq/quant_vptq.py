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
        self.quant_config.output_dir = Path(self.quant_config.output_dir) / self.quant_config.model_name
        for k, v in self.quant_config.layer_config.to_dict().items():
            setattr(self.quant_config, k, v)

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def get_level_order_linear(self, model, model_prefix, dev):
        level_map = {}
        class Catcher(torch.nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module

            def forward(self, *args, **kwargs):
                raise ValueError
        def get_func(name, out_fetures):
            def fake_forward(hidden_state, *args, **kwargs):
                nonlocal level_map
                if len(level_map) == 0:
                    hidden_state *= 0
                key = int(hidden_state[..., 0].item())
                if key not in level_map:
                    level_map[key] = []
                level_map[key].append(name)
                out = torch.ones(hidden_state.shape[:-1]+(out_fetures,), device=hidden_state.device)
                return out*key + 1
            fake_forward.layer_name = name
            return fake_forward

        def fake_forward_2(hidden_state, *args, **kwargs):
            return hidden_state

        attention_layers, pre_layers_of_attention = self.extract_layers(model, model_prefix)
        old_forwards = {}
        for name1,child in attention_layers[0].named_modules():
            old_forwards[name1] = (child, child.forward)
            if len(list(child.modules())) > 1:
                continue
            if not isinstance(child, torch.nn.Linear):
                child.forward = fake_forward_2
            else:
                child.forward = get_func(name1, child.out_features)
        attention_layers[1] = Catcher(attention_layers[1])
        try:  # noqa:SIM105
            model(torch.ones([1, 1], dtype=torch.int64).to(dev))
        except ValueError:
            pass
        attention_layers[1] = attention_layers[1].module
        for _, l_layer in old_forwards.items():
            l_layer[0].forward = l_layer[1]
        return level_map
        
    def collect_hessian_pre(self, model, model_prefix, dev):
        level_linear_names = self.get_level_order_linear(model, model_prefix, "cpu")
        for k in list(level_linear_names.keys()):
            for sub_name in level_linear_names[k]:
                level_linear_names[sub_name] = level_linear_names[k][0]
            level_linear_names.pop(k)
        self.name2hessian = level_linear_names
        if self.quant_config.hessian_path is not None and self.quant_config.inv_hessian_path is not None:
            logger.info("read cached Hessian data")
            _, attention_layers, layer_input_args = self.hijack_block_inputs(model, [(torch.tensor((1, 1), dtype=torch.int64), )], model_prefix, "cpu")
            return attention_layers, layer_input_args
        
        # sample_args = f"--batch_size 2 --devset_size 3072 --ctx_size 8192 \
        #     --base_model {self.quant_config.model_name} --save_path Meta-Llama-3-70B-1 \
        #         --act_save_rate 50 --sample_proc 4"
        from .qllm_hessian import process_collect_hessian
        sample_args = self.quant_config.hessian_config
        sample_args.base_model = self.quant_config.model_name
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
        output = process_collect_hessian(sample_args, embed_func, model, self.tokenizer, level_linear_names, dev)
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
        for i in range(num_gpus):
            output_queue.put(i) # poison pill
        def fetch_next_task(future):
            comm_utils.clear_memory()
            pbar.update(1)
            pbar.set_postfix_str(f'gpu memory: {torch.cuda.memory_allocated(future.gpu_idx)/1024**3:.2f}GB')
            output_queue.put(future.gpu_idx)
            torch.save(future.result(), quant_tmp/f"layer_{future.layer_idx}.pt")

        for layer_idx,layer in enumerate(attention_layers):
            if (quant_tmp/f"layer_{layer_idx}.pt").exists():
                import warnings
                warnings.simplefilter(action='ignore', category=FutureWarning)
                attention_layers[layer_idx] = torch.load(quant_tmp / f"layer_{layer_idx}.pt", weights_only=False)
                pbar.update(1)
                continue
            free_gpu_id = output_queue.get()
            future_task = executor.submit(
                quantize_layer,
                (layer, layer_idx),
                self.quant_config,
                self.quant_config,
                name2hessian=self.name2hessian,
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
        quant_tmp = Path("quant_tmp")
        quant_tmp.mkdir(exist_ok=True)

        if num_gpus > 1:
            self.parallel_quantize(quantize_layer, attention_layers, num_gpus, dev)
        else:        
            for layer_idx in tqdm.trange((len(attention_layers)), desc="running VPTQ"):
                if (quant_tmp/f"layer_{layer_idx}.pt").exists():
                    attention_layers[layer_idx] = torch.load(quant_tmp / f"layer_{layer_idx}.pt", weights_only=False)
                    continue
                attention_layers[layer_idx] = quantize_layer(
                    (attention_layers[layer_idx], layer_idx), self.quant_config, self.quant_config, 
                    name2hessian=self.name2hessian, dev=dev
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
