#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/SmallVector.h>
#include <math.h>
#include "cuda/roi_align_kernel.h"

int roi_align_forward_cuda(int crop_height, int crop_width, float spatial_scale,
                           at::Tensor features, at::Tensor rois, at::Tensor output)
{
    // Grab the input tensor
    float *image_ptr = features.data_ptr<float>();
    float *boxes_ptr = rois.data_ptr<float>();
    float *crops_ptr = output.data_ptr<float>();

    // Number of ROIs
    int num_boxes = rois.size(0);
    int size_rois = rois.size(1);
    if (size_rois != 5)
    {
        return 0;
    }

    // batch size
    int batch = features.size(0);
    // data height
    int image_height = features.size(2);
    // data width
    int image_width = features.size(3);
    // Number of channels
    int depth = features.size(1);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    float extrapolation_value = 0.0;

    ROIAlignForwardLaucher(
        image_ptr, boxes_ptr, num_boxes, batch, image_height, image_width,
        crop_height, crop_width, depth, extrapolation_value, crops_ptr,
        stream);

    return 1;
}

int roi_align_backward_cuda(int crop_height, int crop_width, float spatial_scale,
                            at::Tensor top_grad, at::Tensor rois, at::Tensor bottom_grad)
{
    // Grab the input tensor
    float *grads_ptr = top_grad.data_ptr<float>();
    float *boxes_ptr = rois.data_ptr<float>();
    float *grads_image_ptr = bottom_grad.data_ptr<float>();

    // Number of ROIs
    int num_boxes = rois.size(0);
    int size_rois = rois.size(1);
    if (size_rois != 5)
    {
        return 0;
    }

    // batch size
    int batch = bottom_grad.size(0);
    // data height
    int image_height = bottom_grad.size(2);
    // data width
    int image_width = bottom_grad.size(3);
    // Number of channels
    int depth = bottom_grad.size(1);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    ROIAlignBackwardLaucher(
        grads_ptr, boxes_ptr, num_boxes, batch, image_height, image_width,
        crop_height, crop_width, depth, grads_image_ptr, stream);
    return 1;
}