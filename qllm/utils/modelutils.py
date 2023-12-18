import torch
import torch.nn as nn
import tqdm

DEV = torch.device('cuda:0')

class ScaledLinear(nn.Linear):
    def __init__(self, linear_layer, scale=None):
        super().__init__(linear_layer.in_features, linear_layer.out_features)
        self.scale = nn.Parameter(scale.data)
        self.weight = linear_layer.weight
        self.bias = linear_layer.bias

    def forward(self, x: torch.Tensor):
        x = x.div_(self.scale)
        return nn.functional.linear(x, self.weight, self.bias)

def find_layers(module, layers=[nn.Conv2d, nn.Linear], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(child, layers=layers, name=name + '.' + name1 if name != '' else name1))
    return res


def gen_conditions(_wbits, _groupsize):
    wbits = _wbits
    groupsize = _groupsize
    conditions = []
    while True:
        if wbits >= 8:
            if groupsize == -1 or groupsize == 32:
                break

        if groupsize > 32:
            groupsize /= 2
        else:
            wbits += 1
            groupsize = _groupsize

        conditions.append((int(wbits), int(groupsize)))
    return conditions


def select_quant_linear(pack_mode: str, wbits:int):
    from ..modeling.q_layers import QuantLinear
    from ..modeling.q_layers.quant_linear_awq import WQLinear_GEMM, is_the_machine_support_awq_engine
    from ..modeling.q_layers.quant_linear_onnxruntime import QuantLinearORT

    if pack_mode == "GEMM" or (pack_mode == "AUTO" and is_the_machine_support_awq_engine(wbits)):
        target_layer = WQLinear_GEMM
    elif pack_mode == "ORT":
        target_layer = QuantLinearORT
    else:
        target_layer = QuantLinear
    return target_layer

# copy from https://github.com/openppl-public/ppq/blob/master/ppq/quantization/measure/norm.py
def torch_snr_error(y_pred: torch.Tensor, y_real: torch.Tensor, reduction: str = 'mean') -> torch.Tensor:
    """
    Compute SNR between y_pred(tensor) and y_real(tensor)

    SNR can be calcualted as following equation:

        SNR(pred, real) = (pred - real) ^ 2 / (real) ^ 2

    if x and y are matrixs, SNR error over matrix should be the mean value of SNR error over all elements.

        SNR(pred, real) = mean((pred - real) ^ 2 / (real) ^ 2)
    Args:
        y_pred (torch.Tensor): _description_
        y_real (torch.Tensor): _description_
        reduction (str, optional): _description_. Defaults to 'mean'.
    Raises:
        ValueError: _description_
        ValueError: _description_
    Returns:
        torch.Tensor: _description_
    """
    y_pred = y_pred.type(torch.float32)
    y_real = y_real.type(torch.float32)

    if y_pred.shape != y_real.shape:
        raise ValueError(f'Can not compute snr loss for tensors with different shape. '
                         f'({y_pred.shape} and {y_real.shape})')
    reduction = str(reduction).lower()

    if y_pred.ndim == 1:
        y_pred = y_pred.unsqueeze(0)
        y_real = y_real.unsqueeze(0)

    y_pred = y_pred.flatten(start_dim=1)
    y_real = y_real.flatten(start_dim=1)

    noise_power = torch.pow(y_pred - y_real, 2).sum(dim=-1)
    signal_power = torch.pow(y_real, 2).sum(dim=-1)
    snr = (noise_power) / (signal_power + 1e-7)

    if reduction == 'mean':
        return torch.mean(snr)
    elif reduction == 'sum':
        return torch.sum(snr)
    elif reduction == 'none':
        return snr
    else:
        raise ValueError(f'Unsupported reduction method.')

def get_op_by_name(module, op_name):
    # get the op by its name relative to the module
    for name, m in module.named_modules():
        if name == op_name:
            return m
    raise ValueError(f"Cannot find op {op_name} in module {module}")


def set_op_by_name(layer, name, new_module):
    levels = name.split('.')
    if len(levels) > 1:
        mod_ = layer
        for l_idx in range(len(levels)-1):
            if levels[l_idx].isdigit():
                mod_ = mod_[int(levels[l_idx])]
            else:
                mod_ = getattr(mod_, levels[l_idx])
        setattr(mod_, levels[-1], new_module)
    else:
        setattr(layer, name, new_module)


def get_op_name(module, op):
    # get the name of the op relative to the module
    for name, m in module.named_modules():
        if m is op:
            return name
    raise ValueError(f"Cannot find op {op} in module {module}")


def append_str_prefix(x, prefix):
    if isinstance(x, str):
        return prefix + x
    elif isinstance(x, tuple):
        return tuple([append_str_prefix(y, prefix) for y in x])
    elif isinstance(x, list):
        return [append_str_prefix(y, prefix) for y in x]
    else:
        return x


def make_mixbits_quant_linear(module, replaced_names, quant_info: dict, name='', target_layer=None, device:str="cuda"):
    for module_name, sub_module in tqdm.tqdm(module.named_modules(), total=len(list(module.named_modules())),
                    desc="Replacing linear layers..."):
        if module_name in replaced_names:
            tmp = sub_module
            if "groupsize" in quant_info and 'wbits' in quant_info:
                bits, groupsize = quant_info['wbits'], quant_info['groupsize']
            else:
                bits, groupsize = quant_info[module_name]['wbits'], quant_info[module_name]['groupsize']
            new_module = target_layer(bits, groupsize, tmp.in_features, tmp.out_features, tmp.bias is not None).to(device)
            set_op_by_name(module, module_name, new_module)
    return        
    #if isinstance(module, target_layer):
    #    return
    #for attr in dir(module):
    #    tmp = getattr(module, attr)
    #    name1 = name + '.' + attr if name != '' else attr
    #    if name1 in replaced_names:
    #        delattr(module, attr)
    #        bits, groupsize = quant_info[name1]['wbits'], quant_info[name1]['groupsize']
    #        setattr(module, attr, target_layer(bits, groupsize, tmp.in_features, tmp.out_features, tmp.bias is not None))
    #for name1, child in module.named_children():
    #    make_mixbits_quant_linear(child, replaced_names, quant_info, name + '.' + name1 if name != '' else name1, target_layer)


# deprecated
#def make_quant_linear(module, names, bits, groupsize, name=''):
#    if isinstance(module, QuantLinear):
#        return
#    for attr in dir(module):
#        tmp = getattr(module, attr)
#        name1 = name + '.' + attr if name != '' else attr
#        if name1 in names:
#            delattr(module, attr)
#            setattr(module, attr, QuantLinear(bits, groupsize, tmp.in_features, tmp.out_features, tmp.bias is not None))
#    for name1, child in module.named_children():
#        make_quant_linear(child, names, bits, groupsize, name + '.' + name1 if name != '' else name1)
#
#
#
#def make_linear_qdq_back(module, names, name=''):
#    if isinstance(module, QuantLinear):
#        return
#    for attr in dir(module):
#        tmp = getattr(module, attr)
#        name1 = name + '.' + attr if name != '' else attr
#        if name1 in names:
#            delattr(module, attr)
#            setattr(module, attr, names[name1])
#    for name1, child in module.named_children():
#        make_linear_qdq_back(child, names, name + '.' + name1 if name != '' else name1)