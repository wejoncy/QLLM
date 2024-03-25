from pathlib import Path
import json
from transformers.utils.hub import cached_file
import os
from .. import utils
from ..quantization.config_builder import MetaConfig
logger = utils.logger.get_logger()


class BaseQuantizeConfig:
    def __init__(self):
        self.quant_config = {}
        self.quant_config_by_op = {}
        self.method = None
        self.COMPATIBLE_WITH_AUTOGPTQ = False

    def groupsize(self, layer_name: str = None):
        if layer_name is not None and layer_name in self.quant_config_by_op:
            return self.quant_config_by_op[layer_name]["groupsize"]
        return self.quant_config.get('group_size', None) or self.quant_config.get('q_group_size', None)

    def bits(self, layer_name: str = None):
        if layer_name is not None and layer_name in self.quant_config_by_op:
            return self.quant_config_by_op[layer_name]["wbits"]
        return self.quant_config.get('bits', None) or self.quant_config.get('w_bit', None)

    @property
    def to_meta(self):
        return MetaConfig(self.bits(), self.groupsize(), self.method)

    @property
    def version(self):
        return self.quant_config["version"]

    @version.setter
    def version(self, value):
        self.quant_config["version"] = value

    @staticmethod
    def get_resolved_base_dir(model_name_or_path, quantize_config_filename) -> Path:
        if os.path.isdir(model_name_or_path):  # Local
            resolved_config_file = Path(model_name_or_path) / quantize_config_filename
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
            except:  # noqa : E722
                resolved_config_file = None
        return resolved_config_file

    def try_make_default_quant_op_config(self):
        if self.quant_config_by_op: return
        # backward compatability, we just make a genaral config
        self.quant_config_by_op = {
            "groupsize": self.groupsize(), "wbits": self.bits()}

    def load_quant_op_config(self, model_name_or_path):
        if not (Path(model_name_or_path) / "quant_config_by_layer.json").exists():
            return self.try_make_default_quant_op_config()
        # load quant info
        with open(Path(model_name_or_path) / "quant_config_by_layer.json") as fp:
            qunat_info = json.load(fp)
            self.quant_config_by_op = qunat_info

    def load_quant_config(self, model_name_or_path):
        config_file = self.get_resolved_base_dir(model_name_or_path, "quant_config.json")
        quant_config = None
        if config_file is None:
            # GPTQ-for-llama/AutoGPTQ
            config_file = self.get_resolved_base_dir(model_name_or_path, "quantize_config.json")
        if config_file is not None:
            with open(config_file) as fp:
                quant_config = json.load(fp)

        if config_file is None:
            config_file = self.get_resolved_base_dir(model_name_or_path, "config.json")
        if config_file is not None:
            with open(config_file) as fp:
                quant_config = json.load(fp)
            quant_config = quant_config.get("quantization_config", None)
            assert quant_config.get('use_exllama', False) == False, "use_exllama is not supported yet".

        assert quant_config is not None, ("quant_config.json/quantize_config.json not found in checkpoint directory")

        wbits = quant_config.get("w_bit", quant_config.get("bits", None))
        groupsize = quant_config.get("q_group_size", quant_config.get("group_size", None))
        assert wbits is not None and groupsize is not None

        if quant_config.get('COMPATIBLE_WITH_AUTOGPTQ', None):
            self.COMPATIBLE_WITH_AUTOGPTQ = True
        if "version" not in quant_config:
            self.method = "gptq"
            quant_config["version"] = "GPTQ"
            self.COMPATIBLE_WITH_AUTOGPTQ = True
            import os
            os.environ["COMPATIBLE_WITH_AUTOGPTQ"] = '1'  # FixMe: hacky
        else:  # FIXME is it correct?
            self.method = quant_config.get("method", "awq")
        self.quant_config = quant_config

    @classmethod
    def from_pretrained(cls, model_name_or_path):
        obj = cls()
        obj.load_quant_config(model_name_or_path)
        obj.load_quant_op_config(model_name_or_path)
        return obj
