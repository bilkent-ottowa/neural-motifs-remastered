all: src/cuda/roi_align.cu.o
	python build.py build_ext --inplace
	
src/cuda/roi_align.cu.o: src/cuda/roi_align_kernel.cu
	$(MAKE) -C src/cuda

clean:
	$(MAKE) -C src/cuda clean
