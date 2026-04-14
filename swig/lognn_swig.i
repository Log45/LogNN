%{
#include "tensor.h"
#include "autograd.h"
#include "layers.h"
#include "optim.h"
#include "transformer.h"
#include "losses.h"
#include "model_io.h"
#include "swig_runtime.h"
%}

%include "swig/common_types.i"

%shared_ptr(Module);
%shared_ptr(Linear);
%shared_ptr(Embedding);
%shared_ptr(ReLU_mod);
%shared_ptr(Sigmoid_mod);
%shared_ptr(Tanh_mod);
%shared_ptr(Softmax_mod);
%shared_ptr(Flatten);
%shared_ptr(Dropout);
%shared_ptr(Sequential);
%shared_ptr(LayerNorm);
%shared_ptr(TransformerEncoderLayer);
%shared_ptr(TransformerEncoder);
%shared_ptr(CausalTransformerEncoderLayer);
%shared_ptr(CausalTransformerEncoder);
%shared_ptr(Conv2d);
%shared_ptr(MaxPool2d);
%shared_ptr(AvgPool2d);
%shared_ptr(ConvTranspose2d);
%shared_ptr(BatchNorm2d);

%include "tensor.h"
%include "autograd.h"
%include "layers.h"
%include "optim.h"
%include "transformer.h"
%include "losses.h"
%include "model_io.h"
%rename(is_mlx_native_enabled) backend_mlx_native_available;
%rename(mlx_dispatch_count) backend_mlx_dispatch_count;
%rename(reset_mlx_dispatch_count) backend_mlx_reset_dispatch_count;
%include "swig_runtime.h"
