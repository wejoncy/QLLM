import importlib
import torch

def has_awq_inference_engine():
    return (importlib.util.find_spec("awq_inference_engine") is not None and
            torch.cuda.get_device_properties(0).major * 10+ torch.cuda.get_device_properties(0).minor>=75)


def is_the_machine_support_awq_engine(nbits):
    return has_awq_inference_engine() and nbits == 4

try:
    if importlib.util.find_spec("ort_ops") is not None:
        import ort_ops
        _has_ort_ops = True
except:
    _has_ort_ops = False
    print("no ORT_OPS installed, will skip the running")

def has_ort_ops():
    return _has_ort_ops
