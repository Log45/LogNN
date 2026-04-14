%module(directors="0") lognn_swig

// C# specific conflict fixes and naming polish.
%ignore Tensor::dims;
%ignore Tensor::data;
%ignore Tensor::device;
%ignore Tensor::backend_handle;
%ignore Tensor::total_size;
%ignore Tensor::owns_data;

%rename(ReLU) ReLU_mod;
%rename(Sigmoid) Sigmoid_mod;
%rename(Tanh) Tanh_mod;
%rename(Softmax) Softmax_mod;
%rename(tanh) tanh_act;
%rename(save_model) save_model_binary;
%rename(load_model) load_model_binary;

%include "swig/lognn_swig.i"
