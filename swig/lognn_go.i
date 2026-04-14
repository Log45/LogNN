%module lognn_go

// Go name aliases to stay close to Python surface.
%rename(ReLU) ReLU_mod;
%rename(Sigmoid) Sigmoid_mod;
%rename(Tanh) Tanh_mod;
%rename(Softmax) Softmax_mod;
%rename(tanh) tanh_act;
%rename(save_model) save_model_binary;
%rename(load_model) load_model_binary;

%include "swig/lognn_swig.i"
