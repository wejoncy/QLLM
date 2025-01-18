import torch
import torch.nn as nn
import logging
from ...utils.logger import get_logger

logger = get_logger("qllm")

class InternalVPTQQuantizer(nn.Module):

    def __init__(self):
        super(InternalVPTQQuantizer, self).__init__()


    def quantize_layer(self, tasks, args, quant_args, name2hessian=None, dev=None):
        """
        Quantize the given layers in tasks.
        Args:
            task_id: Task ID
            tasks: List of layers to quantize
            args: Command line arguments
            quant_args: Quantization arguments
            input_queues: Input queue
            output_queues: Output queue
            name2hessian: Dictionary mapping layer names to Hessians
        """
        if dev is not None:
            torch.cuda.set_device(dev)
        try:
            from vptq.layer_quantizer import layer_quantizer
            from vptq.quantize_executer import setup_logging
        except ImportError:
            print("Please install vptq first: pip install vptq")
            raise
            
        layer, layer_idx = tasks
        vptq_logger = setup_logging(f"{args.output_dir}/logs/", str(dev).replace(':', '_'), debug=False)

        dtype = next(iter(layer.parameters())).dtype
        layer, qlinear_args = layer_quantizer(
            args, quant_args, layer, layer_idx, vptq_logger, dev, dtype, name2hessian=name2hessian
        )
        layer = layer.to(dtype).cpu()
        return layer
