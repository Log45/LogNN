#pragma once

#include <functional>
#include <memory>
#include <unordered_set>
#include <vector>

#include "tensor.h"

struct AutoNode {
  Tensor data;
  Tensor grad;
  bool requires_grad;
  std::vector<std::shared_ptr<AutoNode>> parents;
  std::function<void()> backward_fn;

  AutoNode(const Tensor& data, bool requires_grad)
      : data(data),
        grad(Tensor::ones(data.get_dims(), data.get_device_type(), data.get_device_index()).mult(0.0)),
        requires_grad(requires_grad) {}
};

class Variable {
 public:
  std::shared_ptr<AutoNode> node;

  Variable(const Tensor& data, bool requires_grad = false)
      : node(std::make_shared<AutoNode>(data, requires_grad)) {}
  Variable(std::shared_ptr<AutoNode> node) : node(std::move(node)) {}

  Tensor data() const { return node->data; }
  Tensor grad() const { return node->grad; }
  bool requires_grad() const { return node->requires_grad; }
  void set_data(const Tensor& t) { node->data = t; }

  void zero_grad() {
    node->grad = Tensor::ones(node->data.get_dims(), node->data.get_device_type(), node->data.get_device_index()).mult(0.0);
  }

  static Variable add(const Variable& a, const Variable& b);
  static Variable subtract(const Variable& a, const Variable& b);
  static Variable elementwise_mult(const Variable& a, const Variable& b);
  static Variable matmul(const Variable& a, const Variable& b);
  static Variable mult_scalar(const Variable& x, double s);
  static Variable transpose2d(const Variable& x);
  static Variable layer_norm_last_dim(const Variable& x, const Variable& gamma, const Variable& beta,
                                      double eps = 1e-5);
  static Variable relu(const Variable& x);
  static Variable sigmoid(const Variable& x);
  static Variable tanh(const Variable& x);
  static Variable softmax_last_dim(const Variable& x);
  static Variable mean(const Variable& x);
  static Variable mse_loss(const Variable& pred, const Variable& target);

  void backward();

 private:
  static void build_topo(const std::shared_ptr<AutoNode>& cur,
                         std::unordered_set<AutoNode*>& visited,
                         std::vector<std::shared_ptr<AutoNode>>& topo);
};
