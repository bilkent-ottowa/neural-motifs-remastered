#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/SmallVector.h>
#include "highway_lstm_kernel.h"



int highway_lstm_forward_cuda(int inputSize, int hiddenSize, int miniBatch,
        int numLayers, int seqLength,
        at::Tensor x,
        at::Tensor lengths,
        at::Tensor h_data,
        at::Tensor c_data,
        at::Tensor tmp_i,
        at::Tensor tmp_h,
        at::Tensor T,
        at::Tensor bias,
        at::Tensor dropout,
        at::Tensor gates,
        int isTraining) {


    float * x_ptr = x.data_ptr<float>();
    int * lengths_ptr = lengths.data_ptr<int>();
    float * h_data_ptr = h_data.data_ptr<float>();
    float * c_data_ptr = c_data.data_ptr<float>();
    float * tmp_i_ptr = tmp_i.data_ptr<float>();
    float * tmp_h_ptr = tmp_h.data_ptr<float>();
    float * T_ptr = T.data_ptr<float>();
    float * bias_ptr = bias.data_ptr<float>();
    float * dropout_ptr = dropout.data_ptr<float>();
    float * gates_ptr = gates.data_ptr<float>();
    
    if (isTraining == 1) {
        gates_ptr = gates.data_ptr<float>();
    } else {
        gates_ptr = NULL;
    }
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    
    highway_lstm_forward_ongpu(inputSize, hiddenSize, miniBatch, numLayers, 
            seqLength, x_ptr, lengths_ptr, h_data_ptr, c_data_ptr, tmp_i_ptr,
            tmp_h_ptr, T_ptr, bias_ptr, dropout_ptr, gates_ptr,
            isTraining, stream, handle);

    return 1;

}

int highway_lstm_backward_cuda(int inputSize, int hiddenSize, int miniBatch,
        int numLayers, int seqLength,
        at::Tensor out_grad,
        at::Tensor lengths,
        at::Tensor h_data_grad,
        at::Tensor c_data_grad,
        at::Tensor x,
        at::Tensor h_data,
        at::Tensor c_data,
        at::Tensor T,
        at::Tensor gates_out,
        at::Tensor dropout_in,
        at::Tensor h_gates_grad,
        at::Tensor i_gates_grad,
        at::Tensor h_out_grad,
        at::Tensor x_grad,
        at::Tensor T_grad,
        at::Tensor bias_grad,
        int isTraining,
        int do_weight_grad) {

    float * out_grad_ptr = out_grad.data_ptr<float>();
    int * lengths_ptr = lengths.data_ptr<int>();
    float * h_data_grad_ptr = h_data_grad.data_ptr<float>();
    float * c_data_grad_ptr = c_data_grad.data_ptr<float>();
    float * x_ptr = x.data_ptr<float>();
    float * h_data_ptr = h_data.data_ptr<float>();
    float * c_data_ptr = c_data.data_ptr<float>();
    float * T_ptr = T.data_ptr<float>();
    float * gates_out_ptr = gates_out.data_ptr<float>();
    float * dropout_in_ptr = dropout_in.data_ptr<float>();
    float * h_gates_grad_ptr = h_gates_grad.data_ptr<float>();
    float * i_gates_grad_ptr = i_gates_grad.data_ptr<float>();
    float * h_out_grad_ptr = h_out_grad.data_ptr<float>();
    float * x_grad_ptr = x_grad.data_ptr<float>();
    float * T_grad_ptr = T_grad.data_ptr<float>();
    float * bias_grad_ptr = bias_grad.data_ptr<float>();

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();

    highway_lstm_backward_ongpu(inputSize, hiddenSize, miniBatch, numLayers,
            seqLength, out_grad_ptr, lengths_ptr, h_data_grad_ptr, c_data_grad_ptr,
            x_ptr, h_data_ptr, c_data_ptr, T_ptr, gates_out_ptr, dropout_in_ptr,
            h_gates_grad_ptr, i_gates_grad_ptr, h_out_grad_ptr,
            x_grad_ptr, T_grad_ptr, bias_grad_ptr, isTraining, do_weight_grad,
            stream, handle);

    return 1;


}
