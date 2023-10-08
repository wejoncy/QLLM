import io
import os
import re
import subprocess
from typing import List, Set

import torch

from packaging.version import parse, Version
import setuptools
ROOT_DIR = os.path.dirname(__file__)
from torch.utils.cpp_extension import BuildExtension, CUDA_HOME, CUDAExtension


def get_path(*filepath) -> str:
    return os.path.join(ROOT_DIR, *filepath)


def find_version(filepath: str):
    """Extract version information from the given filepath.

    Adapted from https://github.com/ray-project/ray/blob/0b190ee1160eeca9796bc091e07eaebf4c85b511/python/setup.py
    """
    with open(filepath) as fp:
        version_match = re.search(
            r"^__version__ = ['\"]([^'\"]*)['\"]", fp.read(), re.M)
        if version_match:
            return version_match.group(1)
        raise RuntimeError("Unable to find version string.")


def read_readme() -> str:
    """Read the README file."""
    return io.open(get_path("README.md"), "r", encoding="utf-8").read()


def get_requirements() -> List[str]:
    """Get Python package dependencies from requirements.txt."""
    with open(get_path("requirements.txt")) as f:
        requirements = f.read().strip().split("\n")
    return requirements

def get_nvcc_cuda_version(cuda_dir: str) -> Version:
    """Get the CUDA version from nvcc.

    Adapted from https://github.com/NVIDIA/apex/blob/8b7a1ff183741dd8f9b87e7bafd04cfde99cea28/setup.py
    """
    nvcc_output = subprocess.check_output([cuda_dir + "/bin/nvcc", "-V"],
                                          universal_newlines=True)
    output = nvcc_output.split()
    release_idx = output.index("release") + 1
    nvcc_cuda_version = parse(output[release_idx].split(",")[0])
    return nvcc_cuda_version

def get_compute_capabilities():
    # Collect the compute capabilities of all available GPUs.
    compute_capabilities = set()
    for i in range(torch.cuda.device_count()):
        major, minor = torch.cuda.get_device_capability(i)
        if major < 8:
            raise RuntimeError("GPUs with compute capability less than 8.0 are not supported.")
        compute_capabilities.add(major * 10 + minor)

    nvcc_cuda_version = get_nvcc_cuda_version(CUDA_HOME)
    if nvcc_cuda_version < Version("11.1"):
        compute_capabilities.discard(86)
    if nvcc_cuda_version < Version("11.8"):
        compute_capabilities.discard(89)
        compute_capabilities.discard(90)

    capability_flags = []
    for cap in compute_capabilities:
        capability_flags += ["-gencode", f"arch=compute_{cap},code=sm_{cap}"]

    return capability_flags

def get_include_dirs():
    include_dirs = []
    from distutils.sysconfig import get_python_lib

    conda_cuda_include_dir = os.path.join(get_python_lib(), "nvidia/cuda_runtime/include")
    if os.path.isdir(conda_cuda_include_dir):
        include_dirs.append(conda_cuda_include_dir)
    this_dir = os.path.dirname(os.path.abspath(__file__))
    include_dirs.append(this_dir)

    return include_dirs

def get_generator_flag():
    generator_flag = []
    torch_dir = torch.__path__[0]
    if os.path.exists(os.path.join(torch_dir, "include", "ATen", "CUDAGeneratorImpl.h")):
        generator_flag = ["-DOLD_GENERATOR_PATH"]
    
    return generator_flag
if torch.cuda.get_device_properties(0).major >= 8:
    include_dirs = get_include_dirs()
    generator_flags = get_generator_flag()
    arch_flags = get_compute_capabilities()
    if os.name == "nt":
        include_arch = os.getenv("INCLUDE_ARCH", "1") == "1"

        # Relaxed args on Windows
        if include_arch:
            extra_compile_args={"nvcc": arch_flags}
        else:
            extra_compile_args={}
    else:
        extra_compile_args={
            "cxx": ["-g", "-O3", "-fopenmp", "-lgomp", "-std=c++17", "-DENABLE_BF16"],
            "nvcc": [
                "-O3", 
                "-std=c++17",
                "-DENABLE_BF16",
                "-U__CUDA_NO_HALF_OPERATORS__",
                "-U__CUDA_NO_HALF_CONVERSIONS__",
                "-U__CUDA_NO_BFLOAT16_OPERATORS__",
                "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                "-U__CUDA_NO_BFLOAT162_OPERATORS__",
                "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
                "--expt-relaxed-constexpr",
                "--expt-extended-lambda",
                "--use_fast_math",
            ] + arch_flags + generator_flags
        }

    extensions = [
        CUDAExtension(
            "awq_inference_engine",
            [
                "src/awq_cuda/pybind_awq.cpp",
                "src/awq_cuda/quantization/gemm_cuda_gen.cu",
                "src/awq_cuda/layernorm/layernorm.cu",
                "src/awq_cuda/position_embedding/pos_encoding_kernels.cu",
                "src/awq_cuda/quantization/gemv_cuda.cu"
            ], extra_compile_args=extra_compile_args
        )
    ]
else:
    extensions = []


setuptools.setup(
    name="qllm",
    version=find_version(get_path("qllm", "__init__.py")),
    author="qllm Team",
    license="Apache 2.0",
    description="A GPTQ based quantization engine for LLMs",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/wejoncy/GPTQ-for-LLMs",
    project_urls={
        "Homepage": "https://github.com/wejoncy/GPTQ-for-LLMs",
        "Documentation": "https://github.com/wejoncy/GPTQ-for-LLMs",
    },
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    packages=setuptools.find_packages(exclude=("")),
    python_requires=">=3.8",
    install_requires=get_requirements(),
    ext_modules=extensions,
    cmdclass={'build_ext': BuildExtension},
)
