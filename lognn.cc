#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "autograd.h"
#include "layers.h"
#include "losses.h"
#include "optim.h"
#include "tensor.h"
#include "transformer.h"

namespace py = pybind11;

PYBIND11_MODULE(lognn, m) {
  py::class_<Tensor>(m, "Tensor")
      .def(py::init<std::vector<size_t>, std::string, int>(),
           py::arg("dims"), py::arg("device") = "cpu", py::arg("device_index") = 0)
      .def(py::init<std::vector<size_t>, std::vector<std::vector<size_t>>, std::vector<double>, std::string, int>(),
           py::arg("dims"), py::arg("idx"), py::arg("val"),
           py::arg("device") = "cpu", py::arg("device_index") = 0)
      .def_static("ones", &Tensor::ones, py::arg("dims"), py::arg("device") = "cpu", py::arg("device_index") = 0)
      .def_static("zeros", &Tensor::zeros, py::arg("dims"), py::arg("device") = "cpu", py::arg("device_index") = 0)
      .def_static("full", &Tensor::full, py::arg("dims"), py::arg("value"), py::arg("device") = "cpu",
                  py::arg("device_index") = 0)
      .def_static("randn", &Tensor::randn, py::arg("dims"), py::arg("device") = "cpu", py::arg("device_index") = 0,
                  py::arg("seed") = 0)
      .def_static("from_data", &Tensor::from_data, py::arg("dims"), py::arg("values"),
                  py::arg("device") = "cpu", py::arg("device_index") = 0)
      .def("reshape", &Tensor::reshape)
      .def("squeeze", &Tensor::squeeze)
      .def("unsqueeze", &Tensor::unsqueeze, py::arg("dim"))
      .def("transpose", &Tensor::transpose)
      .def("neg", &Tensor::neg)
      .def("reciprocal", &Tensor::reciprocal)
      .def("add", &Tensor::add)
      .def("subtract", &Tensor::subtract)
      .def("mult", &Tensor::mult)
      .def("elementwise_mult", &Tensor::elementwise_mult)
      .def("pow", &Tensor::pow)
      .def("relu", &Tensor::relu)
      .def("sigmoid", &Tensor::sigmoid)
      .def("tanh", &Tensor::tanh)
      .def("softmax_last_dim", &Tensor::softmax_last_dim)
      .def("sum", &Tensor::sum)
      .def("mean", &Tensor::mean)
      .def("exp", &Tensor::exp)
      .def("matmul", &Tensor::matmul)
      .def("get_data", &Tensor::get_data)
      .def("get_dims", &Tensor::get_dims)
      .def("get_device_type", &Tensor::get_device_type)
      .def("get_device_index", &Tensor::get_device_index);

  py::class_<Variable>(m, "Variable")
      .def(py::init<const Tensor&, bool>(), py::arg("data"), py::arg("requires_grad") = false)
      .def("data", &Variable::data)
      .def("grad", &Variable::grad)
      .def("requires_grad", &Variable::requires_grad)
      .def("set_data", &Variable::set_data, py::arg("data"))
      .def("zero_grad", &Variable::zero_grad)
      .def("backward", &Variable::backward)
      .def_static("add", &Variable::add)
      .def_static("subtract", &Variable::subtract)
      .def_static("elementwise_mult", &Variable::elementwise_mult)
      .def_static("matmul", &Variable::matmul)
      .def_static("reshape", &Variable::reshape, py::arg("x"), py::arg("new_dims"))
      .def_static("mult_scalar", &Variable::mult_scalar, py::arg("x"), py::arg("s"))
      .def_static("transpose2d", &Variable::transpose2d)
      .def_static("layer_norm_last_dim", &Variable::layer_norm_last_dim, py::arg("x"), py::arg("gamma"),
                  py::arg("beta"), py::arg("eps") = 1e-5)
      .def_static("relu", &Variable::relu)
      .def_static("sigmoid", &Variable::sigmoid)
      .def_static("tanh", &Variable::tanh)
      .def_static("softmax_last_dim", &Variable::softmax_last_dim)
      .def_static("mean", &Variable::mean)
      .def_static("mse_loss", &Variable::mse_loss)
      .def_static("embedding_gather", &Variable::embedding_gather, py::arg("weight"), py::arg("token_ids"))
      .def_static("cross_entropy_next_token_lm", &Variable::cross_entropy_next_token_lm, py::arg("logits"),
                  py::arg("tokens"))
      .def_static("add_nchw_bias", &Variable::add_nchw_bias, py::arg("y"), py::arg("bias"))
      .def_static("conv2d", &Variable::conv2d, py::arg("x"), py::arg("weight"), py::arg("bias"), py::arg("stride_h"),
                  py::arg("stride_w"), py::arg("pad_h"), py::arg("pad_w"), py::arg("use_bias") = true)
      .def_static("max_pool2d", &Variable::max_pool2d, py::arg("x"), py::arg("kernel_h"), py::arg("kernel_w"),
                  py::arg("stride_h"), py::arg("stride_w"), py::arg("pad_h"), py::arg("pad_w"))
      .def_static("avg_pool2d", &Variable::avg_pool2d, py::arg("x"), py::arg("kernel_h"), py::arg("kernel_w"),
                  py::arg("stride_h"), py::arg("stride_w"), py::arg("pad_h"), py::arg("pad_w"))
      .def_static("conv_transpose2d", &Variable::conv_transpose2d, py::arg("x"), py::arg("weight"),
                  py::arg("stride_h"), py::arg("stride_w"), py::arg("pad_h"), py::arg("pad_w"),
                  py::arg("output_pad_h") = size_t(0), py::arg("output_pad_w") = size_t(0))
      .def_static("log_softmax_last_dim", &Variable::log_softmax_last_dim, py::arg("x"))
      .def_static("cross_entropy_logits", &Variable::cross_entropy_logits, py::arg("logits"), py::arg("target"),
                  py::arg("ignore_index") = -100.0)
      .def_static("batch_norm2d", &Variable::batch_norm2d, py::arg("x"), py::arg("gamma"), py::arg("beta"),
                  py::arg("running_mean"), py::arg("running_var"), py::arg("momentum") = 0.1, py::arg("eps") = 1e-5,
                  py::arg("training") = true);

  py::module_ optim = m.def_submodule("optim");
  py::class_<SGD>(optim, "SGD")
      .def(py::init<std::vector<Variable>, double>(), py::arg("params"), py::arg("lr") = 1e-2)
      .def("step", &SGD::step)
      .def("zero_grad", &SGD::zero_grad);

  py::class_<Adam>(optim, "Adam")
      .def(py::init<std::vector<Variable>, double, double, double, double>(),
           py::arg("params"), py::arg("lr") = 1e-3, py::arg("beta1") = 0.9,
           py::arg("beta2") = 0.999, py::arg("eps") = 1e-8)
      .def("step", &Adam::step)
      .def("zero_grad", &Adam::zero_grad);

  py::class_<AdamW>(optim, "AdamW")
      .def(py::init<std::vector<Variable>, double, double, double, double, double>(),
           py::arg("params"), py::arg("lr") = 1e-3, py::arg("weight_decay") = 0.01,
           py::arg("beta1") = 0.9, py::arg("beta2") = 0.999, py::arg("eps") = 1e-8)
      .def("step", &AdamW::step)
      .def("zero_grad", &AdamW::zero_grad);

  py::module_ nn = m.def_submodule("nn");

  py::class_<Module, std::shared_ptr<Module>>(nn, "Module")
      .def("forward", &Module::forward)
      .def("parameters", &Module::parameters)
      .def("zero_grad", &Module::zero_grad)
      .def("train", &Module::train)
      .def("eval", &Module::eval)
      .def("is_training", &Module::is_training)
      .def("set_training", &Module::set_training, py::arg("training"));

  py::class_<Linear, Module, std::shared_ptr<Linear>>(nn, "Linear")
      .def(py::init<size_t, size_t, std::string, int>(),
           py::arg("in_features"), py::arg("out_features"),
           py::arg("device") = "cpu", py::arg("device_index") = 0)
      .def("forward", &Linear::forward)
      .def("parameters", &Linear::parameters)
      .def("zero_grad", &Linear::zero_grad)
      .def("train", &Linear::train)
      .def("eval", &Linear::eval)
      .def("is_training", &Linear::is_training);

  py::class_<Embedding, Module, std::shared_ptr<Embedding>>(nn, "Embedding")
      .def(py::init<size_t, size_t, std::string, int>(), py::arg("vocab_size"), py::arg("d_model"),
           py::arg("device") = "cpu", py::arg("device_index") = 0)
      .def("forward_from_indices", &Embedding::forward_from_indices, py::arg("token_ids"))
      .def("parameters", &Embedding::parameters)
      .def("zero_grad", &Embedding::zero_grad)
      .def("train", &Embedding::train)
      .def("eval", &Embedding::eval);

  py::class_<ReLU_mod, Module, std::shared_ptr<ReLU_mod>>(nn, "ReLU")
      .def(py::init<>())
      .def("forward", &ReLU_mod::forward)
      .def("parameters", &ReLU_mod::parameters)
      .def("zero_grad", &ReLU_mod::zero_grad)
      .def("train", &ReLU_mod::train)
      .def("eval", &ReLU_mod::eval);

  py::class_<Sigmoid_mod, Module, std::shared_ptr<Sigmoid_mod>>(nn, "Sigmoid")
      .def(py::init<>())
      .def("forward", &Sigmoid_mod::forward)
      .def("parameters", &Sigmoid_mod::parameters)
      .def("train", &Sigmoid_mod::train)
      .def("eval", &Sigmoid_mod::eval);

  py::class_<Tanh_mod, Module, std::shared_ptr<Tanh_mod>>(nn, "Tanh")
      .def(py::init<>())
      .def("forward", &Tanh_mod::forward)
      .def("parameters", &Tanh_mod::parameters)
      .def("train", &Tanh_mod::train)
      .def("eval", &Tanh_mod::eval);

  py::class_<Softmax_mod, Module, std::shared_ptr<Softmax_mod>>(nn, "Softmax")
      .def(py::init<>())
      .def("forward", &Softmax_mod::forward)
      .def("parameters", &Softmax_mod::parameters)
      .def("train", &Softmax_mod::train)
      .def("eval", &Softmax_mod::eval);

  py::class_<Flatten, Module, std::shared_ptr<Flatten>>(nn, "Flatten")
      .def(py::init<>())
      .def("forward", &Flatten::forward)
      .def("parameters", &Flatten::parameters)
      .def("train", &Flatten::train)
      .def("eval", &Flatten::eval);

  py::class_<Dropout, Module, std::shared_ptr<Dropout>>(nn, "Dropout")
      .def(py::init<double, unsigned>(), py::arg("p") = 0.5, py::arg("seed") = 42)
      .def("forward", &Dropout::forward)
      .def("parameters", &Dropout::parameters)
      .def("train", &Dropout::train)
      .def("eval", &Dropout::eval);

  py::class_<Sequential, Module, std::shared_ptr<Sequential>>(nn, "Sequential")
      .def(py::init<std::vector<std::shared_ptr<Module>>>(), py::arg("modules"))
      .def("forward", &Sequential::forward)
      .def("parameters", &Sequential::parameters)
      .def("zero_grad", &Sequential::zero_grad)
      .def("train", &Sequential::train)
      .def("eval", &Sequential::eval);

  py::class_<LayerNorm, Module, std::shared_ptr<LayerNorm>>(nn, "LayerNorm")
      .def(py::init<size_t, std::string, int>(), py::arg("d_model"), py::arg("device") = "cpu",
           py::arg("device_index") = 0)
      .def("forward", &LayerNorm::forward)
      .def("parameters", &LayerNorm::parameters)
      .def("train", &LayerNorm::train)
      .def("eval", &LayerNorm::eval);

  py::class_<TransformerEncoderLayer, Module, std::shared_ptr<TransformerEncoderLayer>>(nn,
                                                                                       "TransformerEncoderLayer")
      .def(py::init<size_t, double, std::string, int>(), py::arg("d_model"), py::arg("dropout") = 0.1,
           py::arg("device") = "cpu", py::arg("device_index") = 0)
      .def("forward", &TransformerEncoderLayer::forward)
      .def("parameters", &TransformerEncoderLayer::parameters)
      .def("zero_grad", &TransformerEncoderLayer::zero_grad)
      .def("train", &TransformerEncoderLayer::train)
      .def("eval", &TransformerEncoderLayer::eval);

  py::class_<TransformerEncoder, Module, std::shared_ptr<TransformerEncoder>>(nn, "TransformerEncoder")
      .def(py::init<size_t, size_t, double, std::string, int>(), py::arg("num_layers"), py::arg("d_model"),
           py::arg("dropout") = 0.1, py::arg("device") = "cpu", py::arg("device_index") = 0)
      .def("forward", &TransformerEncoder::forward)
      .def("parameters", &TransformerEncoder::parameters)
      .def("zero_grad", &TransformerEncoder::zero_grad)
      .def("train", &TransformerEncoder::train)
      .def("eval", &TransformerEncoder::eval);

  py::class_<CausalTransformerEncoderLayer, Module, std::shared_ptr<CausalTransformerEncoderLayer>>(
      nn, "CausalTransformerEncoderLayer")
      .def(py::init<size_t, double, std::string, int>(), py::arg("d_model"), py::arg("dropout") = 0.1,
           py::arg("device") = "cpu", py::arg("device_index") = 0)
      .def("forward", &CausalTransformerEncoderLayer::forward)
      .def("parameters", &CausalTransformerEncoderLayer::parameters)
      .def("zero_grad", &CausalTransformerEncoderLayer::zero_grad)
      .def("train", &CausalTransformerEncoderLayer::train)
      .def("eval", &CausalTransformerEncoderLayer::eval);

  py::class_<CausalTransformerEncoder, Module, std::shared_ptr<CausalTransformerEncoder>>(nn,
                                                                                           "CausalTransformerEncoder")
      .def(py::init<size_t, size_t, double, std::string, int>(), py::arg("num_layers"), py::arg("d_model"),
           py::arg("dropout") = 0.1, py::arg("device") = "cpu", py::arg("device_index") = 0)
      .def("forward", &CausalTransformerEncoder::forward)
      .def("parameters", &CausalTransformerEncoder::parameters)
      .def("zero_grad", &CausalTransformerEncoder::zero_grad)
      .def("train", &CausalTransformerEncoder::train)
      .def("eval", &CausalTransformerEncoder::eval);

  py::class_<Conv2d, Module, std::shared_ptr<Conv2d>>(nn, "Conv2d")
      .def(py::init<size_t, size_t, size_t, size_t, size_t, size_t, size_t, size_t, bool, std::string, int>(),
           py::arg("in_channels"), py::arg("out_channels"), py::arg("kernel_h"), py::arg("kernel_w"),
           py::arg("stride_h") = 1, py::arg("stride_w") = 1, py::arg("pad_h") = 0, py::arg("pad_w") = 0,
           py::arg("bias") = true, py::arg("device") = "cpu", py::arg("device_index") = 0)
      .def("forward", &Conv2d::forward)
      .def("parameters", &Conv2d::parameters)
      .def("zero_grad", &Conv2d::zero_grad)
      .def("train", &Conv2d::train)
      .def("eval", &Conv2d::eval);

  py::class_<MaxPool2d, Module, std::shared_ptr<MaxPool2d>>(nn, "MaxPool2d")
      .def(py::init<size_t, size_t, size_t>(), py::arg("kernel_size"), py::arg("stride") = 0,
           py::arg("padding") = 0)
      .def(py::init<size_t, size_t, size_t, size_t, size_t, size_t>(), py::arg("kernel_h"), py::arg("kernel_w"),
           py::arg("stride_h"), py::arg("stride_w"), py::arg("pad_h"), py::arg("pad_w"))
      .def("forward", &MaxPool2d::forward)
      .def("parameters", &MaxPool2d::parameters)
      .def("train", &MaxPool2d::train)
      .def("eval", &MaxPool2d::eval);

  py::class_<AvgPool2d, Module, std::shared_ptr<AvgPool2d>>(nn, "AvgPool2d")
      .def(py::init<size_t, size_t, size_t>(), py::arg("kernel_size"), py::arg("stride") = 0,
           py::arg("padding") = 0)
      .def(py::init<size_t, size_t, size_t, size_t, size_t, size_t>(), py::arg("kernel_h"), py::arg("kernel_w"),
           py::arg("stride_h"), py::arg("stride_w"), py::arg("pad_h"), py::arg("pad_w"))
      .def("forward", &AvgPool2d::forward)
      .def("parameters", &AvgPool2d::parameters)
      .def("train", &AvgPool2d::train)
      .def("eval", &AvgPool2d::eval);

  py::class_<ConvTranspose2d, Module, std::shared_ptr<ConvTranspose2d>>(nn, "ConvTranspose2d")
      .def(py::init<size_t, size_t, size_t, size_t, size_t, size_t, size_t, size_t, size_t, size_t, bool,
                     std::string, int>(),
           py::arg("in_channels"), py::arg("out_channels"), py::arg("kernel_h"), py::arg("kernel_w"),
           py::arg("stride_h") = 1, py::arg("stride_w") = 1, py::arg("pad_h") = 0, py::arg("pad_w") = 0,
           py::arg("output_pad_h") = 0, py::arg("output_pad_w") = 0, py::arg("bias") = true,
           py::arg("device") = "cpu", py::arg("device_index") = 0)
      .def("forward", &ConvTranspose2d::forward)
      .def("parameters", &ConvTranspose2d::parameters)
      .def("zero_grad", &ConvTranspose2d::zero_grad)
      .def("train", &ConvTranspose2d::train)
      .def("eval", &ConvTranspose2d::eval);

  py::class_<BatchNorm2d, Module, std::shared_ptr<BatchNorm2d>>(nn, "BatchNorm2d")
      .def(py::init<size_t, double, double, std::string, int>(), py::arg("num_features"),
           py::arg("momentum") = 0.1, py::arg("eps") = 1e-5, py::arg("device") = "cpu", py::arg("device_index") = 0)
      .def("forward", &BatchNorm2d::forward)
      .def("parameters", &BatchNorm2d::parameters)
      .def("zero_grad", &BatchNorm2d::zero_grad)
      .def("train", &BatchNorm2d::train)
      .def("eval", &BatchNorm2d::eval)
      .def_readonly("gamma", &BatchNorm2d::gamma)
      .def_readonly("beta", &BatchNorm2d::beta)
      .def_readonly("running_mean", &BatchNorm2d::running_mean)
      .def_readonly("running_var", &BatchNorm2d::running_var);

  m.def("relu", &relu);
  m.def("sigmoid", &sigmoid);
  m.def("tanh", &tanh_act);
  m.def("mse_loss", &mse_loss);
  m.def("save_model", [](const std::shared_ptr<Module>& model, const std::string& path) {
    if (!model) {
      throw std::runtime_error("save_model expects a valid model");
    }
    const auto params = model->parameters();
    py::list tensors;
    for (const auto& p : params) {
      py::dict item;
      const Tensor t = p.data();
      item["dims"] = t.get_dims();
      item["data"] = t.get_data();
      tensors.append(std::move(item));
    }
    py::dict payload;
    payload["format_version"] = 1;
    payload["tensors"] = std::move(tensors);

    py::object pickle = py::module_::import("pickle");
    py::object builtins = py::module_::import("builtins");
    py::object fh = builtins.attr("open")(path, "wb");
    try {
      pickle.attr("dump")(payload, fh, pickle.attr("HIGHEST_PROTOCOL"));
      fh.attr("close")();
    } catch (...) {
      fh.attr("close")();
      throw;
    }
  }, py::arg("model"), py::arg("path"));
  m.def("load_model", [](const std::shared_ptr<Module>& model, const std::string& path) {
    if (!model) {
      throw std::runtime_error("load_model expects a valid model");
    }
    py::object pickle = py::module_::import("pickle");
    py::object builtins = py::module_::import("builtins");
    py::object fh = builtins.attr("open")(path, "rb");
    py::object loaded;
    try {
      loaded = pickle.attr("load")(fh);
      fh.attr("close")();
    } catch (...) {
      fh.attr("close")();
      throw std::runtime_error("load_model failed to unpickle checkpoint");
    }
    py::dict payload;
    try {
      payload = loaded.cast<py::dict>();
    } catch (...) {
      throw std::runtime_error("load_model checkpoint must be a dict");
    }
    if (!payload.contains("format_version") || !payload.contains("tensors")) {
      throw std::runtime_error("load_model checkpoint missing format_version or tensors");
    }

    py::list tensors;
    try {
      tensors = payload["tensors"].cast<py::list>();
    } catch (...) {
      throw std::runtime_error("load_model tensors field must be a list");
    }

    auto params = model->parameters();
    if (tensors.size() != params.size()) {
      throw std::runtime_error("load_model parameter count mismatch");
    }

    for (size_t i = 0; i < params.size(); ++i) {
      py::dict item;
      try {
        item = tensors[i].cast<py::dict>();
      } catch (...) {
        throw std::runtime_error("load_model tensor entry must be a dict");
      }
      if (!item.contains("dims") || !item.contains("data")) {
        throw std::runtime_error("load_model tensor entry missing dims or data");
      }
      const auto current_dims = params[i].data().get_dims();
      const auto load_dims = item["dims"].cast<std::vector<size_t>>();
      if (current_dims != load_dims) {
        throw std::runtime_error("load_model tensor shape mismatch");
      }
    }

    for (size_t i = 0; i < params.size(); ++i) {
      py::dict item = tensors[i].cast<py::dict>();
      const auto dims = item["dims"].cast<std::vector<size_t>>();
      const auto data = item["data"].cast<std::vector<double>>();
      const Tensor cur = params[i].data();
      Tensor restored = Tensor::from_data(dims, data, cur.get_device_type(), cur.get_device_index());
      params[i].set_data(restored);
      params[i].zero_grad();
    }
  }, py::arg("model"), py::arg("path"));
  m.def("is_mlx_native_enabled", &backend_mlx_native_available);
  m.def("mlx_dispatch_count", &backend_mlx_dispatch_count);
  m.def("reset_mlx_dispatch_count", &backend_mlx_reset_dispatch_count);
}
