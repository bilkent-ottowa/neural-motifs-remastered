from setuptools import setup, Distribution
import os
import torch
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension, include_paths
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
extra_objects = ['src/cuda/roi_align_kernel.cu']
extra_objects = [os.path.join(this_file, fname) for fname in extra_objects]
print(sources + extra_objects)
setup(
    name='roi_align_cuda',
    ext_modules=[
        CUDAExtension(
            name='roi_align_cuda',
            sources=sources + extra_objects,
            # extra_objects=extra_objects,
            define_macros=defines,
            include_dirs=[os.path.join(this_file, 'src'), include_paths(),
                          '/home/yigityildirim/OpenAI/OpenAI non-IB/.venv/lib/python3.8/site-packages/torch/include',
                          '/home/yigityildirim/OpenAI/OpenAI non-IB/.venv/lib/python3.8/site-packages/torch/include/torch'],
            extra_compile_args={
                'cxx': ['-g'],
                'nvcc': ['-O2'],
                
            },
            extra_link_args=['-Wl,--no-as-needed', '-lcuda']
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)

