import importlib
import torch
import functools
from ...utils.logger import get_logger
logger = get_logger()

@functools.lru_cache()
def has_package(package_name):
    try:
        if importlib.util.find_spec(package_name) is not None:
            importlib.import_module(package_name)
            return True
    except:  # noqa: E722
        logger.warning(f"Failed to import {package_name}")
    return False


def has_awq_inference_engine():
    return (torch.cuda.get_device_properties(0).major * 10 + torch.cuda.get_device_properties(0).minor >= 75
            and has_package("qllm.awq_inference_engine"))


def is_the_machine_support_awq_engine(nbits):
    return has_awq_inference_engine() and nbits == 4


def has_ort_ops():
    return has_package("qllm.ort_ops")
