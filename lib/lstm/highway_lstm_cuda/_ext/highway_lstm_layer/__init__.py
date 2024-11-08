# from cffi import FFI
# import os

# ffi = FFI()

# # Load the shared library
# lib_path = '/home/yigityildirim/OpenAI/OpenAI non-IB/neural-motifs/lib/lstm/highway_lstm_cuda/_ext/highway_lstm_layer.cpython-38-x86_64-linux-gnu.so'
# lib = ffi.dlopen(lib_path)

# # Define the function signatures
# ffi.cdef("""
# void highway_lstm_forward_ongpu(int inputSize, int hiddenSize, int miniBatch, int numLayers, int seqLength, float *x, int *lengths, float*h_data, float *c_data, float *tmp_i, float *tmp_h, float *T, float *bias, float *dropout, float *gates, int is_training, cudaStream_t stream, cublasHandle_t handle);

# void highway_lstm_backward_ongpu(int inputSize, 
#                                 int hiddenSize, 
#                                 int miniBatch, 
#                                 int numLayers, 
#                                 int seqLength, 
#                                 float *out_grad, 
#                                 int *lengths, 
#                                 float *h_data_grad, 
#                                 float *c_data_grad, 
#                                 float *x, 
#                                 float *h_data, 
#                                 float *c_data, 
#                                 float *T, 
#                                 float *gates_out, 
#                                 float *dropout_in, 
#                                 float *h_gates_grad, 
#                                 float *i_gates_grad, 
#                                 float *h_out_grad, 
#                                 float *x_grad, 
#                                 float *T_grad, 
#                                 float *bias_grad, 
#                                 int isTraining, 
#                                 int do_weight_grad,
#                                 cudaStream_t stream, 
#                                 cublasHandle_t handle);
# """)

# __all__ = []
# def _import_symbols(locals):
#     for symbol in dir(lib):
#         fn = getattr(lib, symbol)
#         locals[symbol] = fn
#         __all__.append(symbol)

# _import_symbols(locals())

# from torch.utils.ffi import _wrap_function
# from ._highway_lstm_layer import lib as _lib, ffi as _ffi

# __all__ = []
# def _import_symbols(locals):
#     for symbol in dir(_lib):
#         fn = getattr(_lib, symbol)
#         locals[symbol] = _wrap_function(fn, _ffi)
#         __all__.append(symbol)

# _import_symbols(locals())
