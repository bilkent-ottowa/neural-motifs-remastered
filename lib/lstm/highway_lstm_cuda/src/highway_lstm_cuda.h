#include <ATen/ATen.h>

int highway_lstm_forward_cuda(int inputSize, int hiddenSize, int miniBatch, int numLayers, int seqLength,
    at::Tensor* *x, at::Tensor* *lengths, at::Tensor* *h_data,
    at::Tensor* *c_data, at::Tensor* *tmp_i,
    at::Tensor* *tmp_h, at::Tensor* *T, at::Tensor* *bias,
    at::Tensor* *dropout, at::Tensor* *gates, int isTraining);

int highway_lstm_backward_cuda(int inputSize, int hiddenSize, int miniBatch, 
        int numLayers, int seqLength, at::Tensor* *out_grad, at::Tensor* *lengths,
        at::Tensor* *h_data_grad, at::Tensor* *c_data_grad, at::Tensor* *x, 
        at::Tensor* *h_data, at::Tensor* *c_data, at::Tensor* *T,
        at::Tensor* *gates_out, at::Tensor* *dropout_in,
        at::Tensor* *h_gates_grad, at::Tensor* *i_gates_grad,
        at::Tensor* *h_out_grad, at::Tensor* *x_grad,  at::Tensor* *T_grad,
        at::Tensor* *bias_grad, int isTraining, int do_weight_grad);
