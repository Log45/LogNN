#pragma once

#include <cmath>
#include <unordered_map>
#include <vector>

#include "autograd.h"

class SGD {
 public:
  explicit SGD(std::vector<Variable> params, double lr = 1e-2)
      : params_(std::move(params)), lr_(lr) {}

  void zero_grad() {
    for (auto& p : params_) p.zero_grad();
  }

  void step() {
    for (auto& p : params_) {
      if (!p.requires_grad()) continue;
      p.set_data(p.data().subtract(p.grad().mult(lr_)));
    }
  }

 private:
  std::vector<Variable> params_;
  double lr_;
};

class Adam {
 public:
  explicit Adam(std::vector<Variable> params, double lr = 1e-3, double beta1 = 0.9, double beta2 = 0.999, double eps = 1e-8)
      : params_(std::move(params)), lr_(lr), beta1_(beta1), beta2_(beta2), eps_(eps), step_(0) {}

  void zero_grad() {
    for (auto& p : params_) p.zero_grad();
  }

  void step() {
    ++step_;
    for (auto& p : params_) {
      if (!p.requires_grad()) continue;
      auto key = p.node.get();
      if (!m_.count(key)) {
        m_.emplace(key, Tensor::ones(p.data().get_dims(), p.data().get_device_type(), p.data().get_device_index()).mult(0.0));
        v_.emplace(key, Tensor::ones(p.data().get_dims(), p.data().get_device_type(), p.data().get_device_index()).mult(0.0));
      }
      Tensor g = p.grad();
      Tensor m_prev = m_.at(key);
      Tensor v_prev = v_.at(key);
      Tensor m = m_prev.mult(beta1_).add(g.mult(1.0 - beta1_));
      Tensor v = v_prev.mult(beta2_).add(g.pow(2.0).mult(1.0 - beta2_));
      m_.at(key) = m;
      v_.at(key) = v;

      const double b1_corr = 1.0 - std::pow(beta1_, static_cast<double>(step_));
      const double b2_corr = 1.0 - std::pow(beta2_, static_cast<double>(step_));
      Tensor mhat = m.mult(1.0 / b1_corr);
      Tensor vhat = v.mult(1.0 / b2_corr);

      std::vector<double> vhost = vhat.get_data();
      for (double& x : vhost) x = std::sqrt(x) + eps_;
      Tensor denom = Tensor::from_data(vhat.get_dims(), vhost, vhat.get_device_type(), vhat.get_device_index());
      Tensor update = mhat.elementwise_mult(denom.reciprocal()).mult(lr_);
      p.set_data(p.data().subtract(update));
    }
  }

 private:
  std::vector<Variable> params_;
  double lr_;
  double beta1_;
  double beta2_;
  double eps_;
  size_t step_;
  std::unordered_map<AutoNode*, Tensor> m_;
  std::unordered_map<AutoNode*, Tensor> v_;
};
