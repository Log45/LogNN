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
      .def("zero_grad", &Variable::zero_grad)
      .def("backward", &Variable::backward)
      .def_static("add", &Variable::add)
      .def_static("subtract", &Variable::subtract)
      .def_static("elementwise_mult", &Variable::elementwise_mult)
      .def_static("matmul", &Variable::matmul)
      .def_static("mult_scalar", &Variable::mult_scalar, py::arg("x"), py::arg("s"))
      .def_static("transpose2d", &Variable::transpose2d)
      .def_static("layer_norm_last_dim", &Variable::layer_norm_last_dim, py::arg("x"), py::arg("gamma"),
                  py::arg("beta"), py::arg("eps") = 1e-5)
      .def_static("relu", &Variable::relu)
      .def_static("sigmoid", &Variable::sigmoid)
      .def_static("tanh", &Variable::tanh)
      .def_static("softmax_last_dim", &Variable::softmax_last_dim)
      .def_static("mean", &Variable::mean)
      .def_static("mse_loss", &Variable::mse_loss);

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

  m.def("relu", &relu);
  m.def("sigmoid", &sigmoid);
  m.def("tanh", &tanh_act);
  m.def("mse_loss", &mse_loss);
}
