#pragma once

#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include "layers.h"

// Layer normalization over the last dimension. x shape [N, D], gamma/beta [1, D].
class LayerNorm : public Module {
 public:
  LayerNorm(size_t d_model, std::string device = "cpu", int device_index = 0)
      : gamma_(Tensor::ones({1, d_model}, device, device_index), true),
        beta_(Tensor::ones({1, d_model}, device, device_index).mult(0.0), true) {}

  Variable forward(const Variable& x) override {
    return Variable::layer_norm_last_dim(x, gamma_, beta_, 1e-5);
  }

  std::vector<Variable> parameters() override { return {gamma_, beta_}; }

 private:
  Variable gamma_;
  Variable beta_;
};

// Single-head encoder layer, pre-norm. Input x shape [T, d_model] (sequence length T, one batch).
class TransformerEncoderLayer : public Module {
 public:
  TransformerEncoderLayer(size_t d_model, double dropout = 0.1, std::string device = "cpu",
                          int device_index = 0)
      : norm1_(d_model, device, device_index),
        norm2_(d_model, device, device_index),
        wq_(d_model, d_model, device, device_index),
        wk_(d_model, d_model, device, device_index),
        wv_(d_model, d_model, device, device_index),
        wo_(d_model, d_model, device, device_index),
        ffn1_(d_model, 4 * d_model, device, device_index),
        ffn2_(4 * d_model, d_model, device, device_index),
        drop_attn_(dropout),
        drop_res_(dropout),
        scale_attn_(1.0 / std::sqrt(static_cast<double>(d_model))) {}

  void set_training(bool training) override {
    training_ = training;
    drop_attn_.set_training(training);
    drop_res_.set_training(training);
  }

  Variable forward(const Variable& x) override {
    Variable n1 = norm1_.forward(x);
    Variable q = wq_.forward(n1);
    Variable k = wk_.forward(n1);
    Variable v = wv_.forward(n1);
    Variable scores = Variable::mult_scalar(Variable::matmul(q, Variable::transpose2d(k)), scale_attn_);
    Variable attn = Variable::softmax_last_dim(scores);
    Variable ctx = Variable::matmul(attn, v);
    Variable proj = wo_.forward(ctx);
    proj = drop_attn_.forward(proj);
    Variable r1 = Variable::add(x, proj);

    Variable n2 = norm2_.forward(r1);
    Variable h = ffn1_.forward(n2);
    Variable h_act = Variable::relu(h);
    Variable ff = ffn2_.forward(h_act);
    ff = drop_res_.forward(ff);
    return Variable::add(r1, ff);
  }

  std::vector<Variable> parameters() override {
    std::vector<Variable> p = norm1_.parameters();
    auto append = [&p](Module& m) {
      auto mp = m.parameters();
      p.insert(p.end(), mp.begin(), mp.end());
    };
    append(norm2_);
    append(wq_);
    append(wk_);
    append(wv_);
    append(wo_);
    append(ffn1_);
    append(ffn2_);
    return p;
  }

 private:
  LayerNorm norm1_;
  LayerNorm norm2_;
  Linear wq_, wk_, wv_, wo_;
  Linear ffn1_, ffn2_;
  Dropout drop_attn_;
  Dropout drop_res_;
  double scale_attn_;
};

class TransformerEncoder : public Module {
 public:
  TransformerEncoder(size_t num_layers, size_t d_model, double dropout = 0.1,
                     std::string device = "cpu", int device_index = 0) {
    for (size_t i = 0; i < num_layers; ++i) {
      layers_.push_back(
          std::make_shared<TransformerEncoderLayer>(d_model, dropout, device, device_index));
    }
  }

  void set_training(bool training) override {
    training_ = training;
    for (auto& L : layers_) L->set_training(training);
  }

  Variable forward(const Variable& x) override {
    Variable y = x;
    for (auto& L : layers_) y = L->forward(y);
    return y;
  }

  std::vector<Variable> parameters() override {
    std::vector<Variable> p;
    for (auto& L : layers_) {
      auto lp = L->parameters();
      p.insert(p.end(), lp.begin(), lp.end());
    }
    return p;
  }

 private:
  std::vector<std::shared_ptr<TransformerEncoderLayer>> layers_;
};
