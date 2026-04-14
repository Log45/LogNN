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

/// Lookup table [vocab, d_model]. Use `forward_from_indices` with token ids tensor [T].
class Embedding : public Module {
 public:
  Variable weight;

  Embedding(size_t vocab, size_t d_model, std::string device = "cpu", int device_index = 0)
      : weight(Tensor::randn({vocab, d_model}, device, device_index, 12345u), true) {}

  Variable forward_from_indices(const Tensor& token_ids_1d) {
    return Variable::embedding_gather(weight, token_ids_1d);
  }

  std::vector<Variable> parameters() override { return {weight}; }
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

class Flatten : public Module {
 public:
  Variable forward(const Variable& x) override {
    const auto d = x.data().get_dims();
    if (d.empty()) {
      throw std::runtime_error("Flatten expects at least 1D input");
    }
    if (d.size() == 1) {
      return Variable::reshape(x, {1, d[0]});
    }
    const size_t n = d[0];
    size_t tail = 1;
    for (size_t i = 1; i < d.size(); ++i) tail *= d[i];
    return Variable::reshape(x, {n, tail});
  }
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

class Conv2d : public Module {
 public:
  Variable weight;
  Variable bias;
  size_t stride_h = 1;
  size_t stride_w = 1;
  size_t pad_h = 0;
  size_t pad_w = 0;
  bool use_bias = true;

  Conv2d(size_t in_channels, size_t out_channels, size_t kernel_h, size_t kernel_w, size_t stride_h_p = 1,
         size_t stride_w_p = 1, size_t pad_h_p = 0, size_t pad_w_p = 0, bool bias = true,
         std::string device = "cpu", int device_index = 0)
      : weight(init_conv_weight(in_channels, out_channels, kernel_h, kernel_w, device, device_index), true),
        bias(Tensor::zeros({1, out_channels, 1, 1}, device, device_index), bias),
        stride_h(stride_h_p),
        stride_w(stride_w_p),
        pad_h(pad_h_p),
        pad_w(pad_w_p),
        use_bias(bias) {}

  Variable forward(const Variable& x) override {
    return Variable::conv2d(x, weight, bias, stride_h, stride_w, pad_h, pad_w, use_bias);
  }

  std::vector<Variable> parameters() override {
    if (use_bias) return {weight, bias};
    return {weight};
  }

 private:
  static Tensor init_conv_weight(size_t in_ch, size_t out_ch, size_t kh, size_t kw, const std::string& device,
                                 int device_index) {
    const size_t n = out_ch * in_ch * kh * kw;
    std::vector<double> values(n);
    const double fan_in = static_cast<double>(in_ch * kh * kw);
    const double scale = std::sqrt(2.0 / fan_in);
    for (size_t i = 0; i < n; ++i) {
      double r = static_cast<double>(std::rand()) / static_cast<double>(RAND_MAX);
      values[i] = (2.0 * r - 1.0) * scale;
    }
    return Tensor::from_data({out_ch, in_ch, kh, kw}, values, device, device_index);
  }
};

class MaxPool2d : public Module {
 public:
  size_t kernel_h;
  size_t kernel_w;
  size_t stride_h;
  size_t stride_w;
  size_t pad_h;
  size_t pad_w;

  MaxPool2d(size_t kernel_size, size_t stride = 0, size_t padding = 0)
      : kernel_h(kernel_size),
        kernel_w(kernel_size),
        stride_h(stride == 0 ? kernel_size : stride),
        stride_w(stride == 0 ? kernel_size : stride),
        pad_h(padding),
        pad_w(padding) {}

  MaxPool2d(size_t kh, size_t kw, size_t sh, size_t sw, size_t ph, size_t pw)
      : kernel_h(kh), kernel_w(kw), stride_h(sh), stride_w(sw), pad_h(ph), pad_w(pw) {}

  Variable forward(const Variable& x) override {
    return Variable::max_pool2d(x, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w);
  }
};

class AvgPool2d : public Module {
 public:
  size_t kernel_h;
  size_t kernel_w;
  size_t stride_h;
  size_t stride_w;
  size_t pad_h;
  size_t pad_w;

  AvgPool2d(size_t kernel_size, size_t stride = 0, size_t padding = 0)
      : kernel_h(kernel_size),
        kernel_w(kernel_size),
        stride_h(stride == 0 ? kernel_size : stride),
        stride_w(stride == 0 ? kernel_size : stride),
        pad_h(padding),
        pad_w(padding) {}

  AvgPool2d(size_t kh, size_t kw, size_t sh, size_t sw, size_t ph, size_t pw)
      : kernel_h(kh), kernel_w(kw), stride_h(sh), stride_w(sw), pad_h(ph), pad_w(pw) {}

  Variable forward(const Variable& x) override {
    return Variable::avg_pool2d(x, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w);
  }
};

class ConvTranspose2d : public Module {
 public:
  Variable weight;
  Variable bias;
  size_t stride_h = 1;
  size_t stride_w = 1;
  size_t pad_h = 0;
  size_t pad_w = 0;
  size_t output_pad_h = 0;
  size_t output_pad_w = 0;
  bool use_bias = true;

  /// weight layout [in_channels, out_channels, kH, kW]
  ConvTranspose2d(size_t in_channels, size_t out_channels, size_t kernel_h, size_t kernel_w, size_t stride_h_p = 1,
                  size_t stride_w_p = 1, size_t pad_h_p = 0, size_t pad_w_p = 0, size_t out_pad_h = 0,
                  size_t out_pad_w = 0, bool bias = true, std::string device = "cpu", int device_index = 0)
      : weight(init_transpose_weight(in_channels, out_channels, kernel_h, kernel_w, device, device_index), true),
        bias(Tensor::zeros({1, out_channels, 1, 1}, device, device_index), bias),
        stride_h(stride_h_p),
        stride_w(stride_w_p),
        pad_h(pad_h_p),
        pad_w(pad_w_p),
        output_pad_h(out_pad_h),
        output_pad_w(out_pad_w),
        use_bias(bias) {}

  Variable forward(const Variable& x) override {
    Variable y = Variable::conv_transpose2d(x, weight, stride_h, stride_w, pad_h, pad_w, output_pad_h, output_pad_w);
    if (!use_bias) return y;
    return Variable::add_nchw_bias(y, bias);
  }

  std::vector<Variable> parameters() override {
    if (use_bias) return {weight, bias};
    return {weight};
  }

 private:
  static Tensor init_transpose_weight(size_t in_ch, size_t out_ch, size_t kh, size_t kw, const std::string& device,
                                      int device_index) {
    const size_t n = in_ch * out_ch * kh * kw;
    std::vector<double> values(n);
    const double fan_in = static_cast<double>(in_ch * kh * kw);
    const double scale = std::sqrt(2.0 / fan_in);
    for (size_t i = 0; i < n; ++i) {
      double r = static_cast<double>(std::rand()) / static_cast<double>(RAND_MAX);
      values[i] = (2.0 * r - 1.0) * scale;
    }
    return Tensor::from_data({in_ch, out_ch, kh, kw}, values, device, device_index);
  }
};

class BatchNorm2d : public Module {
 public:
  Variable gamma;
  Variable beta;
  Tensor running_mean;
  Tensor running_var;
  double momentum;
  double eps;

  BatchNorm2d(size_t num_features, double momentum_p = 0.1, double eps_p = 1e-5, std::string device = "cpu",
              int device_index = 0)
      : gamma(Tensor::ones({1, num_features, 1, 1}, device, device_index), true),
        beta(Tensor::zeros({1, num_features, 1, 1}, device, device_index), true),
        running_mean(Tensor::zeros({1, num_features, 1, 1}, device, device_index)),
        running_var(Tensor::ones({1, num_features, 1, 1}, device, device_index)),
        momentum(momentum_p),
        eps(eps_p) {}

  Variable forward(const Variable& x) override {
    return Variable::batch_norm2d(x, gamma, beta, running_mean, running_var, momentum, eps, is_training());
  }

  std::vector<Variable> parameters() override { return {gamma, beta}; }
};

inline Variable relu(const Variable& x) { return Variable::relu(x); }
inline Variable sigmoid(const Variable& x) { return Variable::sigmoid(x); }
inline Variable tanh_act(const Variable& x) { return Variable::tanh(x); }
