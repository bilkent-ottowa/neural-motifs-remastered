all: roi_align_kernel.cu roi_align_kernel.h
	/usr/local/cuda-12.1/bin/nvcc -c -o roi_align.cu.o roi_align_kernel.cu --gpu-architecture=compute_86 --gpu-code=sm_86 --compiler-options -fPIC 
clean:
	rm roi_align.cu.o
