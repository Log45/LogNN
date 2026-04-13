#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "autograd.h"
#include "layers.h"
#include "losses.h"
#include "optim.h"
#include "tensor.h"

namespace py = pybind11;

PYBIND11_MODULE(lognn, m) {
  py::class_<Tensor>(m, "Tensor")
      .def(py::init<std::vector<size_t>, std::string, int>(),
           py::arg("dims"), py::arg("device") = "cpu", py::arg("device_index") = 0)
      .def(py::init<std::vector<size_t>, std::vector<std::vector<size_t>>, std::vector<double>, std::string, int>(),
           py::arg("dims"), py::arg("idx"), py::arg("val"),
           py::arg("device") = "cpu", py::arg("device_index") = 0)
      .def_static("ones", &Tensor::ones, py::arg("dims"), py::arg("device") = "cpu", py::arg("device_index") = 0)
      .def_static("from_data", &Tensor::from_data, py::arg("dims"), py::arg("values"),
                  py::arg("device") = "cpu", py::arg("device_index") = 0)
      .def("reshape", &Tensor::reshape)
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
      .def_static("relu", &Variable::relu)
      .def_static("sigmoid", &Variable::sigmoid)
      .def_static("tanh", &Variable::tanh)
      .def_static("mean", &Variable::mean)
      .def_static("mse_loss", &Variable::mse_loss);

  py::class_<Linear>(m, "Linear")
      .def(py::init<size_t, size_t, std::string, int>(),
           py::arg("in_features"), py::arg("out_features"),
           py::arg("device") = "cpu", py::arg("device_index") = 0)
      .def("forward", &Linear::forward)
      .def("parameters", &Linear::parameters)
      .def("zero_grad", &Linear::zero_grad);

  py::class_<SGD>(m, "SGD")
      .def(py::init<std::vector<Variable>, double>(), py::arg("params"), py::arg("lr") = 1e-2)
      .def("step", &SGD::step)
      .def("zero_grad", &SGD::zero_grad);

  py::class_<Adam>(m, "Adam")
      .def(py::init<std::vector<Variable>, double, double, double, double>(),
           py::arg("params"), py::arg("lr") = 1e-3, py::arg("beta1") = 0.9,
           py::arg("beta2") = 0.999, py::arg("eps") = 1e-8)
      .def("step", &Adam::step)
      .def("zero_grad", &Adam::zero_grad);

  m.def("relu", &relu);
  m.def("sigmoid", &sigmoid);
  m.def("tanh", &tanh_act);
  m.def("mse_loss", &mse_loss);
}
