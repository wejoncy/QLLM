from setuptools import setup
from torch.utils import cpp_extension
from glob import glob
from pathlib import Path
import os
import urllib.request
import tarfile

CurPath = Path(__file__).parent.absolute()

def download_onnxruntime_headers(version="1.16.3"):
    url = f"https://github.com/microsoft/onnxruntime/releases/download/v{version}/onnxruntime-linux-x64-gpu-{version}.tgz"
    fname = Path(f"build/onnxruntime-headers-{version}.tgz").resolve()
    if not Path(str(fname)[:-4]).exists():
        if not fname.exists():
            fname.parent.mkdir(exist_ok=True)
            print("downloading onnxruntime c++ headers from ", url)
            urllib.request.urlretrieve(url, fname)
        with tarfile.open(fname) as tar:
            tar.extractall(path=fname._str[:-4])
    return os.path.join(fname._str[:-4], os.listdir(fname._str[:-4])[0], 'include')

def creat_ort_extensions():
    extensions = [cpp_extension.CppExtension(
        'onnx_ops',
          sources=glob(str(CurPath)+'/*.cc'),
          include_dirs=[
              str(CurPath)+'/etra_ort_headers',
              download_onnxruntime_headers(),
              "/usr/local/cuda-12.0/targets/x86_64-linux/include",
              #"/usr/local/cuda-11.7/targets/x86_64-linux/include"
            ],
        #extra_compile_args={"cxx": ["-O3", "-std=c++17", "-DUSE_CUDA"]}
        extra_compile_args=['-O0', '-g'],
      ),
      ]
    return extensions


def build_ort_extensions(vllm_version):
    print("build ort extensions")
    setup(name='onnx_ops',
          version=vllm_version,
          ext_modules=creat_ort_extensions(),
          cmdclass={'build_ext': cpp_extension.BuildExtension})

if __name__ == '__main__':
    build_ort_extensions("0.1.0")