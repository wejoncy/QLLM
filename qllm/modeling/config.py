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
        self.method = None
        self.load_from_autogptq = False

    
    def groupsize(self, layer_name: str = None):
        if layer_name is not None and layer_name in self.quantize_op_info:
            return self.quantize_op_info[layer_name]["groupsize"]
        return self.quantize_config.get('group_size',None) or self.quantize_config.get('q_group_size',None)
    
    
    def wbits(self, layer_name:str = None):
        if layer_name is not None and layer_name in self.quantize_op_info:
            return self.quantize_op_info[layer_name]["wbits"]
        return self.quantize_config.get('bits', None) or self.quantize_config.get('w_bit', None)

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
        
    def try_make_default_quant_op_config(self):
        if self.quantize_op_info: return
        # backward compatability, we just make a genaral config
        self.quantize_op_info = {
            "groupsize": self.groupsize(), "wbits": self.wbits()}

    def load_quant_op_config(self, model_name_or_path, args):
        if not (Path(model_name_or_path)/"quant.op.json").exists():
            return self.try_make_default_quant_op_config()
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
            self.method = "GPTQ"
            quant_config["version"] = "GPTQ"
            self.load_from_autogptq = True
            import os
            os.environ['load_from_autogptq'] = '1' # FixMe: hacky
        else: #FIXME is it correct?
            self.method = "AWQ"
        pack_mode = quant_config["version"]

        if args.pack_mode != quant_config["version"]:
            logger.warn(f"pack_mode {args.pack_mode} is not compatiable with checkpoint version" +
                        f", will force to use the checkpoint version {pack_mode}")
            args.pack_mode = pack_mode
        self.quantize_config = quant_config

    @classmethod
    def from_pretrained(cls, model_name_or_path, args):
        obj = cls()
        obj.load_quant_config(model_name_or_path, args)
        obj.load_quant_op_config(model_name_or_path, args)
        return obj
