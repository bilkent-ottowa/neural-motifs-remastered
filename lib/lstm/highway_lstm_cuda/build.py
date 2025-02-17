# pylint: disable=invalid-name
from setuptools import setup, Distribution
import os
import torch
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension, include_paths


if not torch.cuda.is_available():
    raise Exception('HighwayLSTM can only be compiled with CUDA')

this_file = os.path.dirname(os.path.realpath(__file__))
sources = [os.path.join(this_file, 'src/highway_lstm_cuda.cpp')]
headers = [os.path.join(this_file, 'src/highway_lstm_cuda.h')]
defines = [('WITH_CUDA', None)]
with_cuda = True


extra_objects = ['src/highway_lstm_kernel.cu.o']
extra_objects = [os.path.join(this_file, fname) for fname in extra_objects]

# Ensure the _ext directory exists
ext_dir = os.path.join(this_file, '_ext')
os.makedirs(ext_dir, exist_ok=True)
print(ext_dir)

setup(
    name='_ext.highway_lstm_layer',
    ext_modules=[
        CUDAExtension(
            name='_ext.highway_lstm_layer',
            sources=sources,
            extra_objects=extra_objects,
            define_macros=defines,
            include_dirs=[os.path.join(this_file, 'src'), include_paths(),
                          '/home/yigityildirim/OpenAI/OpenAI non-IB/.venv/lib/python3.8/site-packages/torch/include',
                          '/home/yigityildirim/OpenAI/OpenAI non-IB/.venv/lib/python3.8/site-packages/torch/include/torch'],
            extra_compile_args={
                'cxx': ['-std=c++17', '-D_GLIBCXX_USE_CXX11_ABI=1'],
                'nvcc': ['-gencode arch=compute_86,code=sm_86']
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)


