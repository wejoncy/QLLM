import io
import os
import re
import subprocess
from typing import List, Set

import torch

from packaging.version import parse, Version
import setuptools
from torch.utils.cpp_extension import BuildExtension, CUDA_HOME, CUDAExtension

ROOT_DIR = os.path.dirname(__file__)


def is_pypi_build():
    #print(os.environ)
    return os.getenv("PYPI_BUILD", "0") == "1"

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
            VERSION = version_match.group(1)
            cuda_version = str(get_nvcc_cuda_version()).replace('.', '')
            VERSION = VERSION + f"+cu{cuda_version}" if is_pypi_build() else VERSION
            return VERSION
        raise RuntimeError("Unable to find version string.")


def read_readme() -> str:
    """Read the README file."""
    return io.open(get_path("README.md"), "r", encoding="utf-8").read()


def get_requirements() -> List[str]:
    """Get Python package dependencies from requirements.txt."""
    with open(get_path("requirements.txt")) as f:
        requirements = f.read().strip().split("\n")
    requirements = [req for req in requirements if 'https' not in req]
    return requirements

def get_nvcc_cuda_version(cuda_dir: str = "") -> Version:
    """Get the CUDA version from nvcc.

    Adapted from https://github.com/NVIDIA/apex/blob/8b7a1ff183741dd8f9b87e7bafd04cfde99cea28/setup.py
    """
    if cuda_dir == "": cuda_dir = os.getenv("CUDA_HOME", CUDA_HOME)
    CUDA_VERSION = os.getenv("CUDA_VERSION", None)
    if CUDA_VERSION is not None:
        nvcc_cuda_version = CUDA_VERSION
    elif not torch.cuda.is_available():
        nvcc_cuda_version = "0.0"
    else:
        nvcc_output = subprocess.check_output([cuda_dir + "/bin/nvcc", "-V"],
                                              universal_newlines=True)
        output = nvcc_output.split()
        release_idx = output.index("release") + 1
        nvcc_cuda_version = output[release_idx].split(",")[0]
    nvcc_cuda_version = parse(nvcc_cuda_version)
    return nvcc_cuda_version

def get_compute_capabilities(compute_capabilities: Set[int], lower: int = 70):
    # Collect the compute capabilities of all available GPUs.
    if len(compute_capabilities) == 0 and (is_pypi_build() or not torch.cuda.is_available()):
        if lower <= 70:
            compute_capabilities.add(70)
        if lower <= 75:
            compute_capabilities.add(75)
        compute_capabilities.add(80)
        compute_capabilities.add(86)
        compute_capabilities.add(89)

    if len(compute_capabilities) == 0:
        for i in range(torch.cuda.device_count()):
            major, minor = torch.cuda.get_device_capability(i)
            if major*10+minor < lower:
                raise RuntimeError(f"GPUs with compute capability less than {lower} are not supported.")
            compute_capabilities.add(major * 10 + minor)

    if len(compute_capabilities) == 0:
        compute_capabilities.add(70)
        compute_capabilities.add(75)
        compute_capabilities.add(80)
        nvcc_cuda_version = get_nvcc_cuda_version()
        if nvcc_cuda_version > Version("11.1"):
            compute_capabilities.add(86)
        if nvcc_cuda_version > Version("11.8"):
            compute_capabilities.add(89)
            compute_capabilities.add(90)

    print(f"build pacakge for archs: {compute_capabilities}")
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

extensions = []

def build_cuda_extensions():
    if CUDA_HOME is None or os.getenv("EXCLUDE_EXTENTION_FOR_FAST_BUILD"):
        print("we are ignoring all cuda related extensions")
        return []
    else:
        print(f"detect cuda home: {CUDA_HOME}")
    #include_dirs = get_include_dirs()
    def get_extra_compile_args(x_arch_flags = None):
        arch_flags = x_arch_flags if x_arch_flags is not None else get_compute_capabilities(set([]))
        generator_flags = get_generator_flag()
        extra_compile_args={
            "nvcc": [
                    "-O3", 
                    "-std=c++17",
                    "-DENABLE_BF16",
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "-U__CUDA_NO_HALF2_OPERATORS__",
                    "-U__CUDA_NO_BFLOAT16_OPERATORS__",
                    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                    "-U__CUDA_NO_BFLOAT162_OPERATORS__",
                    "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
                    "--expt-relaxed-constexpr",
                    "--expt-extended-lambda",
                    "--use_fast_math",
                ] + generator_flags+arch_flags
        }
        if os.name == "nt":
            extra_compile_args["cxx"] = ["/openmp", "/std:c++17"]
        else:
            extra_compile_args["cxx"] = ["-O3", "-fopenmp", "-lgomp", "-std=c++17", "-DENABLE_BF16"]
            
        return extra_compile_args

    def get_gpu_ver():
        prop =torch.cuda.get_device_properties(0)
        return prop.major * 10 + prop.minor
    if not torch.cuda.is_available() or get_gpu_ver() >= 75 or is_pypi_build():
        arch_flags = get_compute_capabilities(set([]), 75)
        extra_compile_args_awq = get_extra_compile_args(arch_flags)
        extensions.append(
            CUDAExtension(
                "awq_inference_engine",
                [
                    "src/awq_cuda/pybind_awq.cpp",
                    "src/awq_cuda/quantization/gemm_cuda_gen.cu",
                    #"src/awq_cuda/layernorm/layernorm.cu",
                    #"src/awq_cuda/position_embedding/pos_encoding_kernels.cu",
                    "src/awq_cuda/quantization/gemv_cuda.cu"
                ], extra_compile_args=extra_compile_args_awq
            )
        )

    extra_compile_args_ort = get_extra_compile_args()
    extensions.append(CUDAExtension("ort_ops", [
        "src/ort_cuda/ort_ops.cc",
        "src/ort_cuda/dq.cu",
        "src/ort_cuda/dq_gemv.cu",
    ], extra_compile_args=extra_compile_args_ort))
    return extensions


setuptools.setup(
    name="qllm",
    version=find_version(get_path("./qllm/", "__init__.py")),
    author="qllm Team",
    license="Apache 2.0",
    description="A general x-bit quantization engine for LLMs,[2-8] bits, awq/gptq",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/wejoncy/QLLM",
    project_urls={
        "Homepage": "https://github.com/wejoncy/QLLM",
        "Documentation": "https://github.com/wejoncy/QLLM",
    },
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    packages=setuptools.find_packages(exclude=("")),
    python_requires=">=3.8",
    install_requires=get_requirements(),
    ext_modules=build_cuda_extensions(),
    cmdclass={'build_ext': BuildExtension},
)
