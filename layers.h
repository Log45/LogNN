#pragma once

#include <cstdlib>
#include <cmath>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "autograd.h"

class Module {
 public:
  virtual ~Module() = default;
  virtual Variable forward(const Variable& x) {
    (void)x;
    throw std::runtime_error("forward is not implemented on this module");
  }
  virtual std::vector<Variable> parameters() { return {}; }
  virtual void zero_grad() {
    for (auto& p : parameters()) p.zero_grad();
  }

  virtual void set_training(bool training) { training_ = training; }
  void train() { set_training(true); }
  void eval() { set_training(false); }
  bool is_training() const { return training_; }

 protected:
  bool training_ = true;
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

class ReLU_mod : public Module {
 public:
  Variable forward(const Variable& x) override { return Variable::relu(x); }
  std::vector<Variable> parameters() override { return {}; }
};

class Sigmoid_mod : public Module {
 public:
  Variable forward(const Variable& x) override { return Variable::sigmoid(x); }
  std::vector<Variable> parameters() override { return {}; }
};

class Tanh_mod : public Module {
 public:
  Variable forward(const Variable& x) override { return Variable::tanh(x); }
  std::vector<Variable> parameters() override { return {}; }
};

class Sequential : public Module {
 public:
  explicit Sequential(std::vector<std::shared_ptr<Module>> layers)
      : layers_(std::move(layers)) {}

  void set_training(bool training) override {
    training_ = training;
    for (auto& m : layers_) m->set_training(training);
  }

  Variable forward(const Variable& x) override {
    Variable y = x;
    for (auto& m : layers_) y = m->forward(y);
    return y;
  }

  std::vector<Variable> parameters() override {
    std::vector<Variable> p;
    for (auto& m : layers_) {
      auto mp = m->parameters();
      p.insert(p.end(), mp.begin(), mp.end());
    }
    return p;
  }

 private:
  std::vector<std::shared_ptr<Module>> layers_;
};

class Softmax_mod : public Module {
 public:
  Variable forward(const Variable& x) override { return Variable::softmax_last_dim(x); }
  std::vector<Variable> parameters() override { return {}; }
};

class Dropout : public Module {
 public:
  explicit Dropout(double p = 0.5, unsigned seed = 42) : p_(p), gen_(seed) {}

  Variable forward(const Variable& x) override {
    if (!is_training() || p_ <= 0.0) return x;
    Tensor xd = x.node->data;
    const double scale = 1.0 / (1.0 - p_);
    std::bernoulli_distribution dist(1.0 - p_);
    std::vector<double> maskv(xd.total_size);
    for (size_t i = 0; i < xd.total_size; ++i) {
      maskv[i] = dist(gen_) ? scale : 0.0;
    }
    Tensor mask_t =
        Tensor::from_data(xd.get_dims(), maskv, xd.get_device_type(), xd.get_device_index());
    Variable mask_var(mask_t, false);
    return Variable::elementwise_mult(x, mask_var);
  }

  std::vector<Variable> parameters() override { return {}; }

 private:
  double p_;
  std::mt19937 gen_;
};

inline Variable relu(const Variable& x) { return Variable::relu(x); }
inline Variable sigmoid(const Variable& x) { return Variable::sigmoid(x); }
inline Variable tanh_act(const Variable& x) { return Variable::tanh(x); }
