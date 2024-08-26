class FakeArgs:
    def __init__(self, **entries):
        self.quant_method = "gptq"
        self.dataset = "wikitext2"
        self.seed = 0
        self.nsamples = 128
        self.percdamp = 0.01
        self.wbits = 4
        self.groupsize = 128
        self.pack_mode = "AUTO"
        self.act_order = False
        self.true_sequential = False
        self.sym = False
        self.allow_mix_bits = False
        self.static_groups = False
        self.__dict__.update(entries)

    # def __getattr__(self, name):
    #    if name not in self.__dict__:
    #        return None
    #    return self.__dict__[name]
