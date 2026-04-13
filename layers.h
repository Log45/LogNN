#pragma once

#include <cstdlib>
#include <cmath>
#include <memory>
#include <vector>

#include "autograd.h"

class Module {
 public:
  virtual ~Module() = default;
  virtual Variable forward(const Variable& x) = 0;
  virtual std::vector<Variable> parameters() = 0;
  virtual void zero_grad() {
    for (auto& p : parameters()) p.zero_grad();
  }
};

class Linear : public Module {
 public:
  Variable weight;
  Variable bias;

  Linear(size_t in_features, size_t out_features,
         std::string device = "cpu", int device_index = 0)
      : weight(init_weight(in_features, out_features, device, device_index), true),
        bias(Tensor::ones({1, out_features}, device, device_index).mult(0.0), true) {}

  Variable forward(const Variable& x) override {
    auto out = Variable::matmul(x, weight);
    auto out_node = std::make_shared<AutoNode>(
        out.node->data.add_rowwise(bias.node->data),
        out.node->requires_grad || bias.node->requires_grad);
    out_node->parents = {out.node, bias.node};
    out_node->backward_fn = [out_node, out_parent = out.node, b = bias.node]() {
      accumulate_bias_grad(out_node, out_parent, b);
    };
    return Variable(out_node);
  }

  std::vector<Variable> parameters() override { return {weight, bias}; }

 private:
  static Tensor init_weight(size_t in_features, size_t out_features,
                            const std::string& device, int device_index) {
    const size_t n = in_features * out_features;
    std::vector<double> values(n);
    const double scale = 1.0 / std::sqrt(static_cast<double>(in_features));
    for (size_t i = 0; i < n; ++i) {
      double r = static_cast<double>(std::rand()) / static_cast<double>(RAND_MAX);
      values[i] = (2.0 * r - 1.0) * scale;
    }
    return Tensor::from_data({in_features, out_features}, values, device, device_index);
  }

  static void accumulate_bias_grad(const std::shared_ptr<AutoNode>& out_node,
                                   const std::shared_ptr<AutoNode>& out_parent,
                                   const std::shared_ptr<AutoNode>& b_node) {
    if (out_parent->requires_grad) {
      out_parent->grad = out_parent->grad.add(out_node->grad);
    }
    if (b_node->requires_grad) {
      std::vector<double> g = out_node->grad.get_data();
      std::vector<size_t> gd = out_node->grad.get_dims();
      std::vector<double> summed(gd[1], 0.0);
      for (size_t i = 0; i < gd[0]; ++i) {
        for (size_t j = 0; j < gd[1]; ++j) {
          summed[j] += g[i * gd[1] + j];
        }
      }
      Tensor bgrad = Tensor::from_data({1, gd[1]}, summed,
                                       out_node->grad.get_device_type(),
                                       out_node->grad.get_device_index());
      b_node->grad = b_node->grad.add(bgrad);
    }
  }
};

inline Variable relu(const Variable& x) { return Variable::relu(x); }
inline Variable sigmoid(const Variable& x) { return Variable::sigmoid(x); }
inline Variable tanh_act(const Variable& x) { return Variable::tanh(x); }
