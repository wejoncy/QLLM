import dataclasses
from dataclasses import dataclass, asdict
import json
import typing

@dataclass
class MetaConfig:
    bits: int
    group_size: int
    quant_method: str


@dataclass
class MetaInterface:
    to_dict = asdict
    dict = asdict
    @property
    def to_meta(self):
        bits = -1
        if hasattr(self, "bits"):
            bits = self.bits
        elif hasattr(self, "w_bit"):
            bits = self.w_bit
        if hasattr(self, "group_size"):
            group_size = self.group_size
        elif hasattr(self, "q_group_size"):
            group_size = self.q_group_size
        return MetaConfig(bits, group_size, self.quant_method)


@dataclass
class GPTQConfig(MetaInterface):
    damp_percent: float
    group_size: int
    desc_act: str
    bits: int
    sym: bool
    allow_mix_bits: bool
    true_sequential: bool
    static_groups: bool = False
    version: str = ""
    quant_method: str = "gptq"


@dataclass
class AWQConfig(MetaInterface):
    q_group_size: int
    w_bit: int
    zero_point: bool
    version: str = ""
    quant_method: str = "awq"


@dataclass
class HQQConfig(MetaInterface):
    group_size: int
    bits: int
    version: str = ""
    quant_method: str = "hqq"


def dataclass_from_dict(klass, d):
    try:
        fieldtypes = {f.name:f.type for f in dataclasses.fields(klass)}
        return klass(**{f:dataclass_from_dict(fieldtypes[f], d[f]) for f in d})
    except:
        return d # Not a dataclass field

@dataclass
class HessianConfig(MetaInterface):
    batch_size : int = 2
    devset_size : int = 32  # 3072
    iter_size : int = 16
    ctx_size : int = 8192
    chunk_size : int = 256
    base_model : str = None
    act_save_rate : int = 50
    sample_proc : int = 4
    scratch_path: str = None
    save_activations: bool = False
    save_path: str = None

@dataclass
class VPTQLayerConfig(MetaInterface):
    bias: bool = dataclasses.field(default=False)
    enable_norm: bool = dataclasses.field(default=True)
    enable_perm: bool = dataclasses.field(default=True)
    group_num: int = dataclasses.field(default=1)
    outlier_size: int = dataclasses.field(default=0)
    group_size: int = dataclasses.field(default=-1)
    vector_lens: tuple = (-1, 8)
    num_centroids: tuple = (-1, 65536)
    num_res_centroids: tuple = (-1, 256)

@dataclass
class VPTQConfig(MetaInterface):
    model_name: str = dataclasses.field(default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    seq_len: int = dataclasses.field(default=8192)
    quant_step: int = dataclasses.field(default=1)
    percdamp: float = dataclasses.field(default=0.01)
    blocksize: int = dataclasses.field(default=128)
    output_dir: str = dataclasses.field(default="outputs")
    seed: int = dataclasses.field(default=0)
    save_model: bool = dataclasses.field(default=False)
    # disable_actorder: bool = dataclasses.field(default=False)
    hessian_path: typing.Optional[str] = dataclasses.field(default=None)
    inv_hessian_path: typing.Optional[str] = dataclasses.field(default=None)
    num_gpus: int = dataclasses.field(default=1)
    # eval_nsamples: int = dataclasses.field(default=128)
    save_qlinear: bool = dataclasses.field(default=False)
    absorb_perm: bool = dataclasses.field(default=True)
    
    npercent : int = 0
    kmeans_mode : str= "hessian"
    norm_dim : int = 1
    ktol : float =  1e-5
    kiter :int = 100

    hessian_config: HessianConfig = dataclasses.field(default_factory=HessianConfig)
    layer_config: VPTQLayerConfig = dataclasses.field(default_factory=VPTQLayerConfig)
    version: str = ""
    quant_method: str = "vptq"

    @classmethod
    def from_dict(cls, config: dict):
        return dataclass_from_dict(cls, config)

@dataclass
class VPTQInferConfig(MetaInterface):
    group_size: int = 8
    bits: int = 2
    version: str = ""
    quant_method: str = "vptq"
    config_for_layers: typing.Dict[str, dict] = dataclasses.field(default_factory=dict)


def build_config(args):
    if args.quant_method == 'gptq':
        config = GPTQConfig(
            damp_percent=args.percdamp,
            group_size=args.groupsize,
            desc_act=args.act_order,
            bits=args.wbits,
            sym=args.sym,
            allow_mix_bits=args.allow_mix_bits,
            true_sequential=args.true_sequential,
            static_groups=args.static_groups,
        )
    elif args.quant_method == 'awq':
        config = AWQConfig(args.groupsize, args.wbits, not args.sym)
    elif args.quant_method == "hqq":
        config = HQQConfig(args.groupsize, args.wbits)
    elif args.quant_method == "vptq":
        with open(args.quant_config, 'r') as fp:
            dict_config = json.load(fp)
        config = VPTQConfig.from_dict(dict_config)
        config.model_name = args.load + args.model # one of them is empty                                                                                                                                           

    return config
