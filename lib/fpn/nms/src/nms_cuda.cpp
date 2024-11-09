#include <math.h>
#include "cuda/nms_kernel.h"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/SmallVector.h>
#include <torch/extension.h>

cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

int nms_apply(at::Tensor keep, at::Tensor boxes_sorted, const float nms_thresh)
{
    int* keep_data = keep.data_ptr<int>();
    const float* boxes_sorted_data = boxes_sorted.data_ptr<float>();

    const int boxes_num = boxes_sorted.size(0);

    const int devId = boxes_sorted.get_device();

    int numTotalKeep = ApplyNMSGPU(keep_data, boxes_sorted_data, boxes_num, nms_thresh, devId);
    return numTotalKeep;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("nms_apply", &nms_apply, "NMS apply (CUDA)");
}