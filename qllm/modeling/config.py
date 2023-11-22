from pathlib import Path
import json
from transformers.utils.hub import cached_file
import os
from .. import utils

logger = utils.logger.get_logger()


class BaseQuantizeConfig:
    def __init__(self):
        self.args = None
        self.quantize_config = {}
        self.quantize_op_info = {}

    def get_resolved_base_dir(self, model_name_or_path, quantize_config_filename) -> Path:
        if os.path.isdir(model_name_or_path):  # Local
            resolved_config_file = Path(model_name_or_path)/quantize_config_filename
            if not resolved_config_file.exists():
                resolved_config_file = None
        else:  # Remote
            user_agent = {"file_type": "config", "from_auto_class": True}
            try:
                resolved_config_file = cached_file(
                    model_name_or_path,
                    quantize_config_filename,
                    cache_dir=None,
                    user_agent=user_agent,
                )
                resolved_config_file = Path(resolved_config_file)
            except :
                resolved_config_file = None
        return resolved_config_file
        
    def try_make_default_quant_op_config(self, layers, args):
        # backward compatability
        quant_layers_json = {layer_name: {"groupsize": args.groupsize, "wbits": args.wbits}
                                for layer_name in layers.keys() if len(layer_name.split('.')) > 3}
        quant_layers_json["method"] = args.method
        self.quantize_op_info = quant_layers_json

    def load_quant_op_config(self, model_name_or_path, args):
        if not (Path(model_name_or_path)/"quant.op.json").exists():
            return
        # load quant info
        with open(Path(model_name_or_path)/"quant.op.json") as fp:
            qunat_info = json.load(fp)
            args.method = qunat_info["method"]
            args.qunat_info = qunat_info
            self.quantize_op_info = qunat_info



    def load_quant_config(self, model_name_or_path, args):
        if self.get_resolved_base_dir(model_name_or_path, "quant_config.json"):
            config_file = self.get_resolved_base_dir(model_name_or_path, "quant_config.json")
            quant_config = json.load(open(config_file))
            args.wbits = quant_config["w_bit"]
            args.groupsize = quant_config["q_group_size"]
        # GPTQ-for-llama/AutoGPTQ
        elif self.get_resolved_base_dir(model_name_or_path, "quantize_config.json"):
            config_file = self.get_resolved_base_dir(model_name_or_path, "quantize_config.json")
            quant_config = json.load(open(config_file))
            args.wbits = quant_config["bits"]
            args.groupsize = quant_config["group_size"]
        else:
            raise ValueError("quant_config.json not found in checkpoint directory")
        
        if "version" not in quant_config:
            quant_config["version"] = "GPTQ"
            import os
            os.environ['load_from_autogptq'] = '1' # FixMe: hacky
        pack_mode = quant_config["version"]

        if args.pack_mode != quant_config["version"]:
            logger.warn(f"pack_mode {args.pack_mode} is not compatiable with checkpoint version" +
                        f"{pack_mode}, will force to use the checkpoint version {pack_mode}")
            args.pack_mode = pack_mode
        self.quantize_config = quant_config

    def from_pretrained(self, model_name_or_path, args):
        self.load_quant_op_config(model_name_or_path, args)
        self.load_quant_config(model_name_or_path, args)
        return self
