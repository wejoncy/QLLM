from dataclasses import dataclass, asdict


@dataclass
class MetaConfig:
    bits: int
    group_size: int
    method: str


@dataclass
class MetaInterface:
    dict = asdict
    @property
    def to_meta(self):
        if hasattr(self, "bits"):
            bits = self.bits
        elif hasattr(self, "w_bit"):
            bits = self.w_bit
        if hasattr(self, "group_size"):
            group_size = self.group_size
        elif hasattr(self, "q_group_size"):
            group_size = self.q_group_size
        return MetaConfig(bits, group_size, self.method)


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
    method: str = "gptq"


@dataclass
class AWQConfig(MetaInterface):
    q_group_size: int
    w_bit: int
    zero_point: bool
    version: str = ""
    method: str = "awq"


@dataclass
class HQQConfig(MetaInterface):
    group_size: int
    bits: int
    version: str = ""
    method: str = "hqq"


def build_config(args):
    if args.method == 'gptq':
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
    elif args.method == 'awq':
        config = AWQConfig(args.groupsize, args.wbits, not args.sym)
    elif args.method == "hqq":
        config = HQQConfig(args.groupsize, args.wbits)

    return config
