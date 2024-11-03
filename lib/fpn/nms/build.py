from setuptools import setup, Distribution
import os
import torch
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension

# Might have to export PATH=/usr/local/cuda-8.0/bin${PATH:+:${PATH}}

sources = []
headers = []
defines = []
with_cuda = False
this_file = os.path.dirname(os.path.realpath(__file__))

if torch.cuda.is_available():
    print('Including CUDA code.')
    sources += [os.path.join(this_file, 'src/nms_cuda.cpp')]
    headers += [os.path.join(this_file, 'src/nms_cuda.h')]
    defines += [('WITH_CUDA', None)]
    with_cuda = True


print(this_file)
extra_objects = ['src/cuda/nms.cu.o']
extra_objects = [os.path.join(this_file, fname) for fname in extra_objects]

# Ensure the _ext directory exists
ext_dir = os.path.join(this_file, '_ext')
os.makedirs(ext_dir, exist_ok=True)

setup(
    name='_ext.nms',
    ext_modules=[
        CUDAExtension(
            name='_ext.nms',
            sources=sources,
            extra_objects=extra_objects,
            define_macros=defines,
            include_dirs=[os.path.join(this_file, 'src')],
            extra_compile_args={
                'cxx': ['-std=c++17', '-D_GLIBCXX_USE_CXX11_ABI=0'],
                'nvcc': ['-std=c++14']
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)