from setuptools import setup, Distribution
import os
import torch
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension
# Might have to export PATH=/usr/local/cuda-8.0/bin${PATH:+:${PATH}}

# sources = ['src/roi_align.c']
# headers = ['src/roi_align.h']
sources = []
headers = []
defines = []
with_cuda = False
this_file = os.path.dirname(os.path.realpath(__file__))
if torch.cuda.is_available():
    print('Including CUDA code.')
    sources += [os.path.join(this_file, 'src/roi_align_cuda.cpp')]
    headers += [os.path.join(this_file, 'src/roi_align_cuda.h')]
    defines += [('WITH_CUDA', None)]
    with_cuda = True


print(this_file)
extra_objects = ['src/cuda/roi_align.cu.o']
extra_objects = [os.path.join(this_file, fname) for fname in extra_objects]

setup(
    name='_ext.roi_align',
    ext_modules=[
        CUDAExtension(
            name='_ext.roi_align',
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

