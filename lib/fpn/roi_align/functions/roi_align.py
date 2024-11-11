"""
performs ROI aligning
"""

import torch
from torch.autograd import Function
from .._ext import roi_align

class RoIAlignFunction(Function):
    def __init__(self, aligned_height, aligned_width, spatial_scale):
        self.aligned_width = int(aligned_width)
        self.aligned_height = int(aligned_height)
        self.spatial_scale = float(spatial_scale)

        self.feature_size = None

    @staticmethod
    def forward(ctx, features, rois, spatial_scale, aligned_height, aligned_width):
        
        rois_normalized = rois.clone()

        feature_size = features.size()
        batch_size, num_channels, data_height, data_width = feature_size

        height = (data_height -1) / spatial_scale
        width = (data_width - 1) / spatial_scale

        rois_normalized[:,1] /= width
        rois_normalized[:,2] /= height
        rois_normalized[:,3] /= width
        rois_normalized[:,4] /= height


        num_rois = rois.size(0)

        output = features.new(num_rois, num_channels, aligned_height,
            aligned_width).zero_()

        if features.is_cuda:
            res = roi_align.roi_align_forward_cuda(aligned_height,
                                             aligned_width,
                                             spatial_scale, features,
                                             rois_normalized, output)
            assert res == 1
        else:
            raise ValueError

        ctx.save_for_backward(rois, feature_size, aligned_height, aligned_width, spatial_scale)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # assert(self.feature_size is not None and grad_output.is_cuda)

        rois, feature_size, aligned_height, aligned_width, spatial_scale = ctx.saved_tensors
        

        rois_normalized = rois.clone()

        batch_size, num_channels, data_height, data_width = feature_size

        height = (data_height -1) / spatial_scale
        width = (data_width - 1) / spatial_scale

        rois_normalized[:,1] /= width
        rois_normalized[:,2] /= height
        rois_normalized[:,3] /= width
        rois_normalized[:,4] /= height

        grad_input = rois_normalized.new(batch_size, num_channels, data_height,
                                  data_width).zero_()
        res = roi_align.roi_align_backward_cuda(aligned_height,
                                          aligned_width,
                                          spatial_scale, grad_output,
                                          rois_normalized, grad_input)
        assert res == 1
        return grad_input, None
