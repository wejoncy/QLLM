import gc
import torch


def clear_memory(*args):
    for weight in args:
        del weight
    gc.collect()
    torch.cuda.empty_cache()


def get_Model_Size(model):
    param_size = 0
    param_sum = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
        param_sum += param.nelement()
    buffer_size = 0
    buffer_sum = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
        buffer_sum += buffer.nelement()
    all_size = (param_size + buffer_size) / 1024 / 1024
    return all_size


def retrieve_onnx_inputs(model, sample_inputs):
    user_inputs = []

    def hook_for_inputs(mod, inputs, kwargs):
        user_inputs.append((inputs, kwargs))
        return user_inputs[0]
    hook_handle = model.register_forward_pre_hook(hook_for_inputs, with_kwargs=True)
    import inspect
    forward_params = inspect.signature(model.forward).parameters
    input_keys = list(forward_params.keys())
    default_values = [forward_params.get(key).default for key in input_keys]
    model(sample_inputs[0], attention_mask=sample_inputs[1])
    hook_handle.remove()
    user_inputs = user_inputs[0]
    onnx_inputs = default_values
    for idx, val in enumerate(user_inputs[0]):
        onnx_inputs[idx] = user_inputs[0][idx]
    for key, value in user_inputs[1].items():
        idx = input_keys.index(key)
        onnx_inputs[idx] = value
    for value in onnx_inputs:
        if type(value) is torch.Tensor:
            value.to(model.device)
    return input_keys, tuple(onnx_inputs)


def disable_huggingface_init():
    # do not init model twice as it slow initialization
    import torch
    import torch.nn.init
    torch.nn.init.kaiming_uniform_ = lambda x, *args, **kwargs: x
    torch.nn.init.uniform_ = lambda x, *args, **kwargs: x
    torch.nn.init.normal_ = lambda x, *args, **kwargs: x
    torch.nn.init.constant_ = lambda x, *args, **kwargs: x
    torch.nn.init.xavier_uniform_ = lambda x, *args, **kwargs: x
    torch.nn.init.xavier_normal_ = lambda x, *args, **kwargs: x
    torch.nn.init.kaiming_normal_ = lambda x, *args, **kwargs: x
    torch.nn.init.orthogonal_ = lambda x, *args, **kwargs: x
