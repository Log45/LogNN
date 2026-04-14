#include "autograd.h"

#include "conv_impl.h"

#include <cmath>
#include <stdexcept>
#include <vector>

namespace {
void accumulate_grad(std::shared_ptr<AutoNode> n, const Tensor& g) {
  if (!n->requires_grad) return;
  n->grad = n->grad.add(g);
}

struct LayerNormFwd {
  std::vector<double> xhat;
  std::vector<double> mean;
  std::vector<double> inv_std;
  size_t N = 0;
  size_t D = 0;
};

Tensor layer_norm_forward_tensor(const Tensor& x, const Tensor& gamma, const Tensor& beta, double eps,
                                 LayerNormFwd& cache) {
  if (x.get_dims().size() != 2 || gamma.get_dims() != std::vector<size_t>({1, x.get_dims()[1]}) ||
      beta.get_dims() != gamma.get_dims()) {
    throw std::runtime_error("layer_norm expects x [N,D] and gamma,beta [1,D]");
  }
  cache.N = x.get_dims()[0];
  cache.D = x.get_dims()[1];
  const size_t N = cache.N, D = cache.D;
  std::vector<double> xh = x.get_data();
  std::vector<double> ga = gamma.get_data();
  std::vector<double> be = beta.get_data();
  cache.xhat.resize(N * D);
  cache.mean.resize(N);
  cache.inv_std.resize(N);
  std::vector<double> y(N * D);
  for (size_t i = 0; i < N; ++i) {
    double sum = 0.0;
    for (size_t j = 0; j < D; ++j) sum += xh[i * D + j];
    double mu = sum / static_cast<double>(D);
    double var = 0.0;
    for (size_t j = 0; j < D; ++j) {
      double d = xh[i * D + j] - mu;
      var += d * d;
    }
    var /= static_cast<double>(D);
    double rstd = 1.0 / std::sqrt(var + eps);
    cache.mean[i] = mu;
    cache.inv_std[i] = rstd;
    for (size_t j = 0; j < D; ++j) {
      double xhat = (xh[i * D + j] - mu) * rstd;
      cache.xhat[i * D + j] = xhat;
      y[i * D + j] = xhat * ga[j] + be[j];
    }
  }
  return Tensor::from_data(x.get_dims(), y, x.get_device_type(), x.get_device_index());
}
}  // namespace

Variable Variable::add(const Variable& a, const Variable& b) {
  auto out_node = std::make_shared<AutoNode>(
      a.node->data.add(b.node->data),
      a.node->requires_grad || b.node->requires_grad);
  out_node->parents = {a.node, b.node};
  out_node->backward_fn = [out_node, a_node = a.node, b_node = b.node]() {
    accumulate_grad(a_node, out_node->grad);
    accumulate_grad(b_node, out_node->grad);
  };
  return Variable(out_node);
}

Variable Variable::subtract(const Variable& a, const Variable& b) {
  auto out_node = std::make_shared<AutoNode>(
      a.node->data.subtract(b.node->data),
      a.node->requires_grad || b.node->requires_grad);
  out_node->parents = {a.node, b.node};
  out_node->backward_fn = [out_node, a_node = a.node, b_node = b.node]() {
    accumulate_grad(a_node, out_node->grad);
    accumulate_grad(b_node, out_node->grad.neg());
  };
  return Variable(out_node);
}

Variable Variable::elementwise_mult(const Variable& a, const Variable& b) {
  auto out_node = std::make_shared<AutoNode>(
      a.node->data.elementwise_mult(b.node->data),
      a.node->requires_grad || b.node->requires_grad);
  out_node->parents = {a.node, b.node};
  out_node->backward_fn = [out_node, a_node = a.node, b_node = b.node]() {
    accumulate_grad(a_node, out_node->grad.elementwise_mult(b_node->data));
    accumulate_grad(b_node, out_node->grad.elementwise_mult(a_node->data));
  };
  return Variable(out_node);
}

Variable Variable::matmul(const Variable& a, const Variable& b) {
  auto out_node = std::make_shared<AutoNode>(
      a.node->data.matmul(b.node->data),
      a.node->requires_grad || b.node->requires_grad);
  out_node->parents = {a.node, b.node};
  out_node->backward_fn = [out_node, a_node = a.node, b_node = b.node]() {
    accumulate_grad(a_node, out_node->grad.matmul(b_node->data.transpose()));
    accumulate_grad(b_node, a_node->data.transpose().matmul(out_node->grad));
  };
  return Variable(out_node);
}

Variable Variable::reshape(const Variable& x, const std::vector<size_t>& new_dims) {
  auto out_node = std::make_shared<AutoNode>(
      x.node->data.reshape(new_dims),
      x.node->requires_grad);
  out_node->parents = {x.node};
  out_node->backward_fn = [out_node, x_node = x.node]() {
    if (!x_node->requires_grad) return;
    accumulate_grad(x_node, out_node->grad.reshape(x_node->data.get_dims()));
  };
  return Variable(out_node);
}

Variable Variable::mult_scalar(const Variable& x, double s) {
  auto out_node = std::make_shared<AutoNode>(
      x.node->data.mult(s),
      x.node->requires_grad);
  out_node->parents = {x.node};
  out_node->backward_fn = [out_node, x_node = x.node, s]() {
    accumulate_grad(x_node, out_node->grad.mult(s));
  };
  return Variable(out_node);
}

Variable Variable::transpose2d(const Variable& x) {
  if (x.node->data.get_dims().size() != 2) {
    throw std::runtime_error("transpose2d expects 2D tensor");
  }
  auto out_node = std::make_shared<AutoNode>(
      x.node->data.transpose(),
      x.node->requires_grad);
  out_node->parents = {x.node};
  out_node->backward_fn = [out_node, x_node = x.node]() {
    accumulate_grad(x_node, out_node->grad.transpose());
  };
  return Variable(out_node);
}

Variable Variable::layer_norm_last_dim(const Variable& x, const Variable& gamma, const Variable& beta,
                                       double eps) {
  auto cache = std::make_shared<LayerNormFwd>();
  Tensor y = layer_norm_forward_tensor(x.node->data, gamma.node->data, beta.node->data, eps, *cache);
  bool req = x.node->requires_grad || gamma.node->requires_grad || beta.node->requires_grad;
  auto out_node = std::make_shared<AutoNode>(y, req);
  out_node->parents = {x.node, gamma.node, beta.node};
  out_node->backward_fn = [out_node, x_node = x.node, g_node = gamma.node, b_node = beta.node, cache]() {
    const size_t N = cache->N, D = cache->D;
    std::vector<double> dout = out_node->grad.get_data();
    std::vector<double> ga = g_node->data.get_data();
    std::vector<double> d_gamma(D, 0.0), d_beta(D, 0.0);
    std::vector<double> dxhat(N * D);
    for (size_t i = 0; i < N; ++i) {
      for (size_t j = 0; j < D; ++j) {
        size_t idx = i * D + j;
        d_gamma[j] += dout[idx] * cache->xhat[idx];
        d_beta[j] += dout[idx];
        dxhat[idx] = dout[idx] * ga[j];
      }
    }
    std::vector<double> dx(N * D);
    const double invd = 1.0 / static_cast<double>(D);
    for (size_t i = 0; i < N; ++i) {
      double sum_dxhat = 0.0, sum_dxhat_xhat = 0.0;
      for (size_t j = 0; j < D; ++j) {
        size_t idx = i * D + j;
        sum_dxhat += dxhat[idx];
        sum_dxhat_xhat += dxhat[idx] * cache->xhat[idx];
      }
      double rstd = cache->inv_std[i];
      for (size_t j = 0; j < D; ++j) {
        size_t idx = i * D + j;
        double val = D * dxhat[idx] - sum_dxhat - cache->xhat[idx] * sum_dxhat_xhat;
        dx[idx] = rstd * invd * val;
      }
    }
    if (x_node->requires_grad) {
      accumulate_grad(x_node, Tensor::from_data({N, D}, dx, x_node->data.get_device_type(), x_node->data.get_device_index()));
    }
    if (g_node->requires_grad) {
      accumulate_grad(g_node, Tensor::from_data({1, D}, d_gamma, g_node->data.get_device_type(), g_node->data.get_device_index()));
    }
    if (b_node->requires_grad) {
      accumulate_grad(b_node, Tensor::from_data({1, D}, d_beta, b_node->data.get_device_type(), b_node->data.get_device_index()));
    }
  };
  return Variable(out_node);
}

Variable Variable::relu(const Variable& x) {
  auto out_node = std::make_shared<AutoNode>(
      x.node->data.relu(),
      x.node->requires_grad);
  out_node->parents = {x.node};
  out_node->backward_fn = [out_node, x_node = x.node]() {
    Tensor mask = x_node->data.binarilize();
    accumulate_grad(x_node, out_node->grad.elementwise_mult(mask));
  };
  return Variable(out_node);
}

Variable Variable::sigmoid(const Variable& x) {
  auto out_node = std::make_shared<AutoNode>(
      x.node->data.sigmoid(),
      x.node->requires_grad);
  out_node->parents = {x.node};
  out_node->backward_fn = [out_node, x_node = x.node]() {
    Tensor one = Tensor::ones(out_node->data.get_dims(), out_node->data.get_device_type(), out_node->data.get_device_index());
    Tensor s = out_node->data;
    Tensor ds = s.elementwise_mult(one.subtract(s));
    accumulate_grad(x_node, out_node->grad.elementwise_mult(ds));
  };
  return Variable(out_node);
}

Variable Variable::tanh(const Variable& x) {
  auto out_node = std::make_shared<AutoNode>(
      x.node->data.tanh(),
      x.node->requires_grad);
  out_node->parents = {x.node};
  out_node->backward_fn = [out_node, x_node = x.node]() {
    Tensor one = Tensor::ones(out_node->data.get_dims(), out_node->data.get_device_type(), out_node->data.get_device_index());
    Tensor ds = one.subtract(out_node->data.pow(2.0));
    accumulate_grad(x_node, out_node->grad.elementwise_mult(ds));
  };
  return Variable(out_node);
}

// Row softmax backward: dx_i = y_i * (g_i - sum_j y_j * g_j) per row.
Variable Variable::softmax_last_dim(const Variable& x) {
  Tensor y = x.node->data.softmax_last_dim();
  auto out_node = std::make_shared<AutoNode>(y, x.node->requires_grad);
  out_node->parents = {x.node};
  out_node->backward_fn = [out_node, x_node = x.node]() {
    std::vector<size_t> d = out_node->data.get_dims();
    if (d.size() != 2) return;
    size_t rows = d[0], cols = d[1];
    std::vector<double> yh = out_node->data.get_data();
    std::vector<double> gh = out_node->grad.get_data();
    std::vector<double> dx(rows * cols);
    for (size_t r = 0; r < rows; ++r) {
      double dot = 0.0;
      for (size_t j = 0; j < cols; ++j) {
        size_t idx = r * cols + j;
        dot += yh[idx] * gh[idx];
      }
      for (size_t j = 0; j < cols; ++j) {
        size_t idx = r * cols + j;
        dx[idx] = yh[idx] * (gh[idx] - dot);
      }
    }
    Tensor g = Tensor::from_data(d, dx, x_node->data.get_device_type(), x_node->data.get_device_index());
    accumulate_grad(x_node, g);
  };
  return Variable(out_node);
}

Variable Variable::mean(const Variable& x) {
  auto out_node = std::make_shared<AutoNode>(
      x.node->data.mean(),
      x.node->requires_grad);
  out_node->parents = {x.node};
  out_node->backward_fn = [out_node, x_node = x.node]() {
    const double g = out_node->grad.get_data()[0] / static_cast<double>(x_node->data.total_size);
    Tensor back = Tensor::ones(x_node->data.get_dims(), x_node->data.get_device_type(), x_node->data.get_device_index()).mult(g);
    accumulate_grad(x_node, back);
  };
  return Variable(out_node);
}

Variable Variable::mse_loss(const Variable& pred, const Variable& target) {
  Variable d = subtract(pred, target);
  Variable sq = elementwise_mult(d, d);
  return mean(sq);
}

Variable Variable::embedding_gather(const Variable& weight, const Tensor& token_ids) {
  if (token_ids.get_dims().size() != 1) {
    throw std::runtime_error("embedding_gather expects token_ids shape [T]");
  }
  const size_t T = token_ids.get_dims()[0];
  if (weight.node->data.get_dims().size() != 2) {
    throw std::runtime_error("embedding_gather expects weight [V, D]");
  }
  const size_t V = weight.node->data.get_dims()[0];
  const size_t D = weight.node->data.get_dims()[1];
  std::vector<double> idx = token_ids.get_data();
  std::vector<double> wh = weight.node->data.get_data();
  std::vector<double> out(T * D);
  for (size_t t = 0; t < T; ++t) {
    const size_t id = static_cast<size_t>(idx[t]);
    if (id >= V) {
      throw std::runtime_error("embedding_gather token id out of range");
    }
    for (size_t d = 0; d < D; ++d) {
      out[t * D + d] = wh[id * D + d];
    }
  }
  Tensor out_t = Tensor::from_data({T, D}, out, weight.node->data.get_device_type(), weight.node->data.get_device_index());
  auto out_node = std::make_shared<AutoNode>(out_t, weight.node->requires_grad);
  out_node->parents = {weight.node};
  out_node->backward_fn = [out_node, w_node = weight.node, token_ids, V, D, T]() {
    if (!w_node->requires_grad) return;
    std::vector<double> idx = token_ids.get_data();
    std::vector<double> gh = out_node->grad.get_data();
    std::vector<double> gw(V * D, 0.0);
    for (size_t t = 0; t < T; ++t) {
      const size_t id = static_cast<size_t>(idx[t]);
      for (size_t d = 0; d < D; ++d) {
        gw[id * D + d] += gh[t * D + d];
      }
    }
    accumulate_grad(w_node, Tensor::from_data({V, D}, gw, w_node->data.get_device_type(), w_node->data.get_device_index()));
  };
  return Variable(out_node);
}

struct CELMCache {
  size_t T = 0;
  size_t V = 0;
  std::vector<double> softmax;  // (T-1)*V for rows 0..T-2
  std::vector<int> targets;     // length T-1, token at position i+1
};

Variable Variable::cross_entropy_next_token_lm(const Variable& logits, const Tensor& tokens_T) {
  if (logits.node->data.get_dims().size() != 2) {
    throw std::runtime_error("cross_entropy_next_token_lm expects logits [T, V]");
  }
  if (tokens_T.get_dims().size() != 1) {
    throw std::runtime_error("cross_entropy_next_token_lm expects tokens [T]");
  }
  const size_t T = logits.node->data.get_dims()[0];
  const size_t V = logits.node->data.get_dims()[1];
  if (tokens_T.get_dims()[0] != T || T < 2) {
    throw std::runtime_error("cross_entropy_next_token_lm: len(tokens) must match T and T>=2");
  }
  auto cache = std::make_shared<CELMCache>();
  cache->T = T;
  cache->V = V;
  const size_t n = T - 1;
  cache->softmax.resize(n * V);
  cache->targets.resize(n);
  std::vector<double> xh = logits.node->data.get_data();
  std::vector<double> th = tokens_T.get_data();
  double loss_sum = 0.0;
  for (size_t i = 0; i < n; ++i) {
    const size_t t_next = static_cast<size_t>(th[i + 1]);
    if (t_next >= V) {
      throw std::runtime_error("cross_entropy_next_token_lm: target token id out of range");
    }
    cache->targets[i] = static_cast<int>(t_next);
    double mx = xh[i * V];
    for (size_t j = 1; j < V; ++j) {
      mx = std::max(mx, xh[i * V + j]);
    }
    double s = 0.0;
    for (size_t j = 0; j < V; ++j) {
      const double e = std::exp(xh[i * V + j] - mx);
      cache->softmax[i * V + j] = e;
      s += e;
    }
    for (size_t j = 0; j < V; ++j) {
      cache->softmax[i * V + j] /= s;
    }
    loss_sum -= std::log(std::max(cache->softmax[i * V + t_next], 1e-30));
  }
  const double mean_loss = loss_sum / static_cast<double>(n);
  Tensor loss_t = Tensor::from_data({1}, {mean_loss}, logits.node->data.get_device_type(), logits.node->data.get_device_index());
  auto out_node = std::make_shared<AutoNode>(loss_t, logits.node->requires_grad);
  out_node->parents = {logits.node};
  out_node->backward_fn = [out_node, logits_node = logits.node, cache]() {
    if (!logits_node->requires_grad) return;
    const size_t T = cache->T;
    const size_t V = cache->V;
    const size_t n = T - 1;
    const double g_up = out_node->grad.get_data()[0];
    const double scale = g_up / static_cast<double>(n);
    std::vector<double> dx(T * V, 0.0);
    for (size_t i = 0; i < n; ++i) {
      const int tgt = cache->targets[i];
      for (size_t j = 0; j < V; ++j) {
        const double sm = cache->softmax[i * V + j];
        const double delta = (static_cast<int>(j) == tgt) ? 1.0 : 0.0;
        dx[i * V + j] = scale * (sm - delta);
      }
    }
    accumulate_grad(logits_node, Tensor::from_data({T, V}, dx, logits_node->data.get_device_type(), logits_node->data.get_device_index()));
  };
  return Variable(out_node);
}

namespace {

size_t idx_nchw(size_t n, size_t c, size_t h, size_t w, size_t C, size_t H, size_t W) {
  return ((n * C + c) * H + h) * W + w;
}

struct MaxPool2dCache {
  MaxPool2dFwdState pool_st;
  size_t N = 0, C = 0, H = 0, W = 0, Ho = 0, Wo = 0, kH = 0, kW = 0, sh = 0, sw = 0, ph = 0, pw = 0;
};

struct AvgPool2dParams {
  size_t N = 0, C = 0, H = 0, W = 0, Ho = 0, Wo = 0, kH = 0, kW = 0, sh = 0, sw = 0, ph = 0, pw = 0;
};

struct ConvTranspose2dCache {
  size_t N = 0, Ci = 0, Hi = 0, Wi = 0, Co = 0, kH = 0, kW = 0, sh = 0, sw = 0, ph = 0, pw = 0, oph = 0, opw = 0,
         Ho = 0, Wo = 0;
};

struct LogSoftmaxCache {
  size_t N = 0, C = 0;
  std::vector<double> softmax;
};

struct CELogitsCache {
  size_t N = 0, C = 0;
  size_t n_valid = 0;
  std::vector<double> softmax;
  std::vector<int> tgt;  // class index or -1 if ignored
};

struct BN2dCache {
  size_t N = 0, C = 0, H = 0, W = 0;
  bool training = true;
  std::vector<double> xhat;
  std::vector<double> mean_c;
  std::vector<double> inv_std_c;
  std::vector<double> gamma_h;
};

}  // namespace

Variable Variable::add_nchw_bias(const Variable& y, const Variable& bias) {
  Tensor::ensure_same_device(y.node->data, bias.node->data, "add_nchw_bias");
  const auto d = y.node->data.get_dims();
  if (d.size() != 4) {
    throw std::runtime_error("add_nchw_bias: y must be NCHW");
  }
  const size_t C = d[1];
  if (bias.node->data.get_dims() != std::vector<size_t>({1, C, 1, 1})) {
    throw std::runtime_error("add_nchw_bias: bias must be [1, C, 1, 1]");
  }
  std::vector<double> yh = y.node->data.get_data();
  std::vector<double> bh = bias.node->data.get_data();
  const size_t N = d[0], H = d[2], W = d[3];
  for (size_t n = 0; n < N; ++n) {
    for (size_t c = 0; c < C; ++c) {
      for (size_t h = 0; h < H; ++h) {
        for (size_t w = 0; w < W; ++w) {
          yh[idx_nchw(n, c, h, w, C, H, W)] += bh[c];
        }
      }
    }
  }
  Tensor out_t = Tensor::from_data(d, yh, y.node->data.get_device_type(), y.node->data.get_device_index());
  const bool req = y.node->requires_grad || bias.node->requires_grad;
  auto out_node = std::make_shared<AutoNode>(out_t, req);
  out_node->parents = {y.node, bias.node};
  out_node->backward_fn = [out_node, y_n = y.node, b_n = bias.node, N, C, H, W]() {
    std::vector<double> g = out_node->grad.get_data();
    if (y_n->requires_grad) {
      accumulate_grad(y_n, Tensor::from_data({N, C, H, W}, g, y_n->data.get_device_type(), y_n->data.get_device_index()));
    }
    if (b_n->requires_grad) {
      std::vector<double> db(C, 0.0);
      for (size_t n = 0; n < N; ++n) {
        for (size_t c = 0; c < C; ++c) {
          for (size_t h = 0; h < H; ++h) {
            for (size_t w = 0; w < W; ++w) {
              db[c] += g[idx_nchw(n, c, h, w, C, H, W)];
            }
          }
        }
      }
      accumulate_grad(b_n, Tensor::from_data({1, C, 1, 1}, db, b_n->data.get_device_type(), b_n->data.get_device_index()));
    }
  };
  return Variable(out_node);
}

Variable Variable::conv2d(const Variable& x, const Variable& weight, const Variable& bias, size_t stride_h,
                          size_t stride_w, size_t pad_h, size_t pad_w, bool use_bias) {
  Tensor::ensure_same_device(x.node->data, weight.node->data, "conv2d");
  const auto xd = x.node->data.get_dims();
  const auto wd = weight.node->data.get_dims();
  if (xd.size() != 4 || wd.size() != 4) {
    throw std::runtime_error("conv2d expects x [N,C,H,W] and weight [C_out,C_in,kH,kW]");
  }
  const size_t Co = wd[0];
  if (use_bias) {
    Tensor::ensure_same_device(x.node->data, bias.node->data, "conv2d bias");
    const auto bd = bias.node->data.get_dims();
    if (bd != std::vector<size_t>({1, Co, 1, 1})) {
      throw std::runtime_error("conv2d bias must be [1, C_out, 1, 1]");
    }
  }
  size_t Ho = 0, Wo = 0;
  Tensor y0 = conv2d_forward(x.node->data, weight.node->data, stride_h, stride_w, pad_h, pad_w, Ho, Wo);
  std::vector<double> yh = y0.get_data();
  const size_t N = xd[0];
  if (use_bias) {
    std::vector<double> bh = bias.node->data.get_data();
    for (size_t n = 0; n < N; ++n) {
      for (size_t c = 0; c < Co; ++c) {
        for (size_t h = 0; h < Ho; ++h) {
          for (size_t w = 0; w < Wo; ++w) {
            yh[idx_nchw(n, c, h, w, Co, Ho, Wo)] += bh[c];
          }
        }
      }
    }
  }
  Tensor y = Tensor::from_data({N, Co, Ho, Wo}, yh, x.node->data.get_device_type(), x.node->data.get_device_index());
  const bool req = x.node->requires_grad || weight.node->requires_grad || (use_bias && bias.node->requires_grad);
  auto out_node = std::make_shared<AutoNode>(y, req);
  out_node->parents = {x.node, weight.node};
  if (use_bias) {
    out_node->parents.push_back(bias.node);
  }
  out_node->backward_fn = [out_node, x_n = x.node, w_n = weight.node, b_n = bias.node, use_bias, stride_h,
                             stride_w, pad_h, pad_w, Ho, Wo]() {
    std::vector<double> dy = out_node->grad.get_data();
    const auto gd = out_node->grad.get_dims();
    const size_t N = gd[0], Co = gd[1];
    if (use_bias && b_n->requires_grad) {
      std::vector<double> db(Co, 0.0);
      for (size_t n = 0; n < N; ++n) {
        for (size_t c = 0; c < Co; ++c) {
          for (size_t h = 0; h < Ho; ++h) {
            for (size_t w = 0; w < Wo; ++w) {
              db[c] += dy[idx_nchw(n, c, h, w, Co, Ho, Wo)];
            }
          }
        }
      }
      accumulate_grad(b_n, Tensor::from_data({1, Co, 1, 1}, db, b_n->data.get_device_type(), b_n->data.get_device_index()));
    }
    Tensor dy_t =
        Tensor::from_data({N, Co, Ho, Wo}, dy, out_node->grad.get_device_type(), out_node->grad.get_device_index());
    auto gxgw = conv2d_backward(dy_t, x_n->data, w_n->data, stride_h, stride_w, pad_h, pad_w, Ho, Wo);
    accumulate_grad(x_n, gxgw.first);
    accumulate_grad(w_n, gxgw.second);
  };
  return Variable(out_node);
}

Variable Variable::max_pool2d(const Variable& x, size_t kernel_h, size_t kernel_w, size_t stride_h, size_t stride_w,
                             size_t pad_h, size_t pad_w) {
  const auto xd = x.node->data.get_dims();
  if (xd.size() != 4) {
    throw std::runtime_error("max_pool2d expects NCHW 4D tensor");
  }
  const size_t N = xd[0], C = xd[1], H = xd[2], W = xd[3];
  auto cache = std::make_shared<MaxPool2dCache>();
  size_t Ho = 0, Wo = 0;
  Tensor y = maxpool2d_forward_tensor(x.node->data, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w,
                                        cache->pool_st, Ho, Wo);
  cache->N = N;
  cache->C = C;
  cache->H = H;
  cache->W = W;
  cache->Ho = Ho;
  cache->Wo = Wo;
  cache->kH = kernel_h;
  cache->kW = kernel_w;
  cache->sh = stride_h;
  cache->sw = stride_w;
  cache->ph = pad_h;
  cache->pw = pad_w;
  auto out_node = std::make_shared<AutoNode>(y, x.node->requires_grad);
  out_node->parents = {x.node};
  out_node->backward_fn = [out_node, x_n = x.node, cache]() {
    if (!x_n->requires_grad) return;
    Tensor dx = maxpool2d_backward_tensor(out_node->grad, cache->pool_st, cache->N, cache->C, cache->H, cache->W,
                                          cache->kH, cache->kW, cache->sh, cache->sw, cache->ph, cache->pw, cache->Ho,
                                          cache->Wo);
    accumulate_grad(x_n, dx);
  };
  return Variable(out_node);
}

Variable Variable::avg_pool2d(const Variable& x, size_t kernel_h, size_t kernel_w, size_t stride_h, size_t stride_w,
                              size_t pad_h, size_t pad_w) {
  const auto xd = x.node->data.get_dims();
  if (xd.size() != 4) {
    throw std::runtime_error("avg_pool2d expects NCHW 4D tensor");
  }
  const size_t N = xd[0], C = xd[1], H = xd[2], W = xd[3];
  size_t Ho = 0, Wo = 0;
  Tensor y = avgpool2d_forward_tensor(x.node->data, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, Ho, Wo);
  auto params = std::make_shared<AvgPool2dParams>();
  params->N = N;
  params->C = C;
  params->H = H;
  params->W = W;
  params->Ho = Ho;
  params->Wo = Wo;
  params->kH = kernel_h;
  params->kW = kernel_w;
  params->sh = stride_h;
  params->sw = stride_w;
  params->ph = pad_h;
  params->pw = pad_w;
  auto out_node = std::make_shared<AutoNode>(y, x.node->requires_grad);
  out_node->parents = {x.node};
  out_node->backward_fn = [out_node, x_n = x.node, params]() {
    if (!x_n->requires_grad) return;
    Tensor dx = avgpool2d_backward_tensor(out_node->grad, params->N, params->C, params->H, params->W, params->kH,
                                          params->kW, params->sh, params->sw, params->ph, params->pw, params->Ho,
                                          params->Wo);
    accumulate_grad(x_n, dx);
  };
  return Variable(out_node);
}

Variable Variable::conv_transpose2d(const Variable& x, const Variable& weight, size_t stride_h, size_t stride_w,
                                     size_t pad_h, size_t pad_w, size_t output_pad_h, size_t output_pad_w) {
  Tensor::ensure_same_device(x.node->data, weight.node->data, "conv_transpose2d");
  const auto xd = x.node->data.get_dims();
  const auto wd = weight.node->data.get_dims();
  if (xd.size() != 4 || wd.size() != 4) {
    throw std::runtime_error("conv_transpose2d expects 4D tensors");
  }
  const size_t N = xd[0], Ci = xd[1], Hi = xd[2], Wi = xd[3];
  const size_t Ci_w = wd[0], Co = wd[1], kH = wd[2], kW = wd[3];
  if (Ci_w != Ci) {
    throw std::runtime_error("conv_transpose2d: weight in_channels must match x channels");
  }
  size_t Ho = 0, Wo = 0;
  Tensor y = conv_transpose2d_forward(x.node->data, weight.node->data, stride_h, stride_w, pad_h, pad_w,
                                        output_pad_h, output_pad_w, Ho, Wo);
  auto cache = std::make_shared<ConvTranspose2dCache>();
  cache->N = N;
  cache->Ci = Ci;
  cache->Hi = Hi;
  cache->Wi = Wi;
  cache->Co = Co;
  cache->kH = kH;
  cache->kW = kW;
  cache->sh = stride_h;
  cache->sw = stride_w;
  cache->ph = pad_h;
  cache->pw = pad_w;
  cache->oph = output_pad_h;
  cache->opw = output_pad_w;
  cache->Ho = Ho;
  cache->Wo = Wo;
  const bool req = x.node->requires_grad || weight.node->requires_grad;
  auto out_node = std::make_shared<AutoNode>(y, req);
  out_node->parents = {x.node, weight.node};
  out_node->backward_fn = [out_node, x_n = x.node, w_n = weight.node, cache]() {
    auto gxgw = conv_transpose2d_backward(out_node->grad, x_n->data, w_n->data, cache->sh, cache->sw, cache->ph,
                                          cache->pw, cache->oph, cache->opw, cache->Ho, cache->Wo);
    accumulate_grad(x_n, gxgw.first);
    accumulate_grad(w_n, gxgw.second);
  };
  return Variable(out_node);
}

Variable Variable::log_softmax_last_dim(const Variable& x) {
  const auto d = x.node->data.get_dims();
  if (d.size() != 2) {
    throw std::runtime_error("log_softmax_last_dim expects [N, C]");
  }
  const size_t N = d[0], C = d[1];
  auto cache = std::make_shared<LogSoftmaxCache>();
  cache->N = N;
  cache->C = C;
  cache->softmax.resize(N * C);
  std::vector<double> xh = x.node->data.get_data();
  std::vector<double> yh(N * C);
  for (size_t i = 0; i < N; ++i) {
    double mx = xh[i * C];
    for (size_t j = 1; j < C; ++j) {
      mx = std::max(mx, xh[i * C + j]);
    }
    double s = 0.0;
    for (size_t j = 0; j < C; ++j) {
      const double e = std::exp(xh[i * C + j] - mx);
      cache->softmax[i * C + j] = e;
      s += e;
    }
    for (size_t j = 0; j < C; ++j) {
      cache->softmax[i * C + j] /= s;
    }
    const double log_s = std::log(std::max(s, 1e-30)) + mx;
    for (size_t j = 0; j < C; ++j) {
      yh[i * C + j] = xh[i * C + j] - log_s;
    }
  }
  Tensor y = Tensor::from_data({N, C}, yh, x.node->data.get_device_type(), x.node->data.get_device_index());
  auto out_node = std::make_shared<AutoNode>(y, x.node->requires_grad);
  out_node->parents = {x.node};
  out_node->backward_fn = [out_node, x_n = x.node, cache]() {
    if (!x_n->requires_grad) return;
    std::vector<double> gh = out_node->grad.get_data();
    const size_t N = cache->N, C = cache->C;
    std::vector<double> dx(N * C);
    for (size_t i = 0; i < N; ++i) {
      double sum_g = 0.0;
      for (size_t j = 0; j < C; ++j) {
        sum_g += gh[i * C + j];
      }
      for (size_t j = 0; j < C; ++j) {
        dx[i * C + j] = gh[i * C + j] - cache->softmax[i * C + j] * sum_g;
      }
    }
    accumulate_grad(x_n, Tensor::from_data({N, C}, dx, x_n->data.get_device_type(), x_n->data.get_device_index()));
  };
  return Variable(out_node);
}

Variable Variable::cross_entropy_logits(const Variable& logits, const Tensor& target, double ignore_index) {
  const auto ld = logits.node->data.get_dims();
  if (ld.size() != 2) {
    throw std::runtime_error("cross_entropy_logits expects logits [N, C]");
  }
  const size_t N = ld[0], C = ld[1];
  if (target.get_dims().size() != 1 || target.get_dims()[0] != N) {
    throw std::runtime_error("cross_entropy_logits expects target [N]");
  }
  auto cache = std::make_shared<CELogitsCache>();
  cache->N = N;
  cache->C = C;
  cache->softmax.resize(N * C);
  cache->tgt.resize(N, -1);
  std::vector<double> xh = logits.node->data.get_data();
  std::vector<double> th = target.get_data();
  double loss_sum = 0.0;
  size_t n_valid = 0;
  for (size_t i = 0; i < N; ++i) {
    const double t_raw = th[i];
    if (std::abs(t_raw - ignore_index) < 1e-15) {
      for (size_t j = 0; j < C; ++j) {
        cache->softmax[i * C + j] = 0.0;
      }
      cache->tgt[i] = -1;
      continue;
    }
    ++n_valid;
    const size_t t = static_cast<size_t>(t_raw);
    if (t >= C) {
      throw std::runtime_error("cross_entropy_logits: target class out of range");
    }
    cache->tgt[i] = static_cast<int>(t);
    double mx = xh[i * C];
    for (size_t j = 1; j < C; ++j) {
      mx = std::max(mx, xh[i * C + j]);
    }
    double s = 0.0;
    for (size_t j = 0; j < C; ++j) {
      const double e = std::exp(xh[i * C + j] - mx);
      cache->softmax[i * C + j] = e;
      s += e;
    }
    for (size_t j = 0; j < C; ++j) {
      cache->softmax[i * C + j] /= s;
    }
    loss_sum -= std::log(std::max(cache->softmax[i * C + t], 1e-30));
  }
  if (n_valid == 0) {
    throw std::runtime_error("cross_entropy_logits: no valid targets");
  }
  cache->n_valid = n_valid;
  const double mean_loss = loss_sum / static_cast<double>(n_valid);
  Tensor loss_t =
      Tensor::from_data({1}, {mean_loss}, logits.node->data.get_device_type(), logits.node->data.get_device_index());
  auto out_node = std::make_shared<AutoNode>(loss_t, logits.node->requires_grad);
  out_node->parents = {logits.node};
  out_node->backward_fn = [out_node, logits_n = logits.node, cache]() {
    if (!logits_n->requires_grad) return;
    const size_t N = cache->N, C = cache->C;
    const size_t n_valid = cache->n_valid;
    const double g_up = out_node->grad.get_data()[0];
    const double scale = g_up / static_cast<double>(n_valid);
    std::vector<double> dx(N * C, 0.0);
    for (size_t i = 0; i < N; ++i) {
      const int t = cache->tgt[i];
      if (t < 0) continue;
      for (size_t j = 0; j < C; ++j) {
        const double delta = (static_cast<int>(j) == t) ? 1.0 : 0.0;
        dx[i * C + j] = scale * (cache->softmax[i * C + j] - delta);
      }
    }
    accumulate_grad(logits_n, Tensor::from_data({N, C}, dx, logits_n->data.get_device_type(), logits_n->data.get_device_index()));
  };
  return Variable(out_node);
}

Variable Variable::batch_norm2d(const Variable& x, const Variable& gamma, const Variable& beta, Tensor& running_mean,
                                Tensor& running_var, double momentum, double eps, bool training) {
  Tensor::ensure_same_device(x.node->data, gamma.node->data, "batch_norm2d");
  Tensor::ensure_same_device(x.node->data, beta.node->data, "batch_norm2d");
  Tensor::ensure_same_device(x.node->data, running_mean, "batch_norm2d running_mean");
  Tensor::ensure_same_device(x.node->data, running_var, "batch_norm2d running_var");
  const auto xd = x.node->data.get_dims();
  if (xd.size() != 4) {
    throw std::runtime_error("batch_norm2d expects NCHW");
  }
  const size_t N = xd[0], C = xd[1], H = xd[2], W = xd[3];
  const size_t S = N * H * W;
  if (gamma.node->data.get_dims() != std::vector<size_t>({1, C, 1, 1}) ||
      beta.node->data.get_dims() != gamma.node->data.get_dims()) {
    throw std::runtime_error("batch_norm2d: gamma and beta must be [1, C, 1, 1]");
  }
  if (running_mean.get_dims() != std::vector<size_t>({1, C, 1, 1}) ||
      running_var.get_dims() != running_mean.get_dims()) {
    throw std::runtime_error("batch_norm2d: running buffers must be [1, C, 1, 1]");
  }
  auto cache = std::make_shared<BN2dCache>();
  cache->N = N;
  cache->C = C;
  cache->H = H;
  cache->W = W;
  cache->training = training;
  cache->xhat.resize(N * C * H * W);
  cache->mean_c.resize(C);
  cache->inv_std_c.resize(C);
  std::vector<double> xh = x.node->data.get_data();
  std::vector<double> gh = gamma.node->data.get_data();
  std::vector<double> bh = beta.node->data.get_data();
  cache->gamma_h = gh;
  std::vector<double> rm = running_mean.get_data();
  std::vector<double> rv = running_var.get_data();
  std::vector<double> yh(N * C * H * W);
  for (size_t c = 0; c < C; ++c) {
    const double g_c = gh[c];
    const double b_c = bh[c];
    double mean = 0.0;
    if (training) {
      for (size_t n = 0; n < N; ++n) {
        for (size_t h = 0; h < H; ++h) {
          for (size_t w = 0; w < W; ++w) {
            mean += xh[idx_nchw(n, c, h, w, C, H, W)];
          }
        }
      }
      mean /= static_cast<double>(S);
      double var = 0.0;
      for (size_t n = 0; n < N; ++n) {
        for (size_t h = 0; h < H; ++h) {
          for (size_t w = 0; w < W; ++w) {
            const double d = xh[idx_nchw(n, c, h, w, C, H, W)] - mean;
            var += d * d;
          }
        }
      }
      var /= static_cast<double>(S);
      const double inv_std = 1.0 / std::sqrt(var + eps);
      cache->mean_c[c] = mean;
      cache->inv_std_c[c] = inv_std;
      rm[c] = (1.0 - momentum) * rm[c] + momentum * mean;
      rv[c] = (1.0 - momentum) * rv[c] + momentum * var;
      for (size_t n = 0; n < N; ++n) {
        for (size_t h = 0; h < H; ++h) {
          for (size_t w = 0; w < W; ++w) {
            const size_t ix = idx_nchw(n, c, h, w, C, H, W);
            const double xhat = (xh[ix] - mean) * inv_std;
            cache->xhat[ix] = xhat;
            yh[ix] = xhat * g_c + b_c;
          }
        }
      }
    } else {
      const double mean = rm[c];
      const double inv_std = 1.0 / std::sqrt(rv[c] + eps);
      cache->mean_c[c] = mean;
      cache->inv_std_c[c] = inv_std;
      for (size_t n = 0; n < N; ++n) {
        for (size_t h = 0; h < H; ++h) {
          for (size_t w = 0; w < W; ++w) {
            const size_t ix = idx_nchw(n, c, h, w, C, H, W);
            const double xhat = (xh[ix] - mean) * inv_std;
            cache->xhat[ix] = xhat;
            yh[ix] = xhat * g_c + b_c;
          }
        }
      }
    }
  }
  running_mean = Tensor::from_data({1, C, 1, 1}, rm, running_mean.get_device_type(), running_mean.get_device_index());
  running_var = Tensor::from_data({1, C, 1, 1}, rv, running_var.get_device_type(), running_var.get_device_index());
  Tensor y = Tensor::from_data({N, C, H, W}, yh, x.node->data.get_device_type(), x.node->data.get_device_index());
  const bool req = x.node->requires_grad || gamma.node->requires_grad || beta.node->requires_grad;
  auto out_node = std::make_shared<AutoNode>(y, req);
  out_node->parents = {x.node, gamma.node, beta.node};
  out_node->backward_fn = [out_node, x_n = x.node, g_n = gamma.node, b_n = beta.node, cache]() {
    const size_t N = cache->N, C = cache->C, H = cache->H, W = cache->W;
    const size_t S = N * H * W;
    std::vector<double> dout = out_node->grad.get_data();
    std::vector<double> d_gamma(C, 0.0), d_beta(C, 0.0);
    std::vector<double> dxhat(N * C * H * W);
    for (size_t c = 0; c < C; ++c) {
      for (size_t n = 0; n < N; ++n) {
        for (size_t h = 0; h < H; ++h) {
          for (size_t w = 0; w < W; ++w) {
            const size_t ix = idx_nchw(n, c, h, w, C, H, W);
            const double dg = dout[ix];
            d_gamma[c] += dg * cache->xhat[ix];
            d_beta[c] += dg;
            dxhat[ix] = dg * cache->gamma_h[c];
          }
        }
      }
    }
    std::vector<double> dx(N * C * H * W);
    if (cache->training) {
      for (size_t c = 0; c < C; ++c) {
        double sum_dxhat = 0.0, sum_dxhat_xhat = 0.0;
        for (size_t n = 0; n < N; ++n) {
          for (size_t h = 0; h < H; ++h) {
            for (size_t w = 0; w < W; ++w) {
              const size_t ix = idx_nchw(n, c, h, w, C, H, W);
              sum_dxhat += dxhat[ix];
              sum_dxhat_xhat += dxhat[ix] * cache->xhat[ix];
            }
          }
        }
        const double inv_std = cache->inv_std_c[c];
        const double inv_s = 1.0 / static_cast<double>(S);
        for (size_t n = 0; n < N; ++n) {
          for (size_t h = 0; h < H; ++h) {
            for (size_t w = 0; w < W; ++w) {
              const size_t ix = idx_nchw(n, c, h, w, C, H, W);
              const double val =
                  static_cast<double>(S) * dxhat[ix] - sum_dxhat - cache->xhat[ix] * sum_dxhat_xhat;
              dx[ix] = inv_std * inv_s * val;
            }
          }
        }
      }
    } else {
      for (size_t c = 0; c < C; ++c) {
        const double inv_std = cache->inv_std_c[c];
        for (size_t n = 0; n < N; ++n) {
          for (size_t h = 0; h < H; ++h) {
            for (size_t w = 0; w < W; ++w) {
              const size_t ix = idx_nchw(n, c, h, w, C, H, W);
              dx[ix] = dxhat[ix] * inv_std;
            }
          }
        }
      }
    }
    if (x_n->requires_grad) {
      accumulate_grad(x_n, Tensor::from_data({N, C, H, W}, dx, x_n->data.get_device_type(), x_n->data.get_device_index()));
    }
    if (g_n->requires_grad) {
      accumulate_grad(g_n, Tensor::from_data({1, C, 1, 1}, d_gamma, g_n->data.get_device_type(), g_n->data.get_device_index()));
    }
    if (b_n->requires_grad) {
      accumulate_grad(b_n, Tensor::from_data({1, C, 1, 1}, d_beta, b_n->data.get_device_type(), b_n->data.get_device_index()));
    }
  };
  return Variable(out_node);
}

void Variable::build_topo(const std::shared_ptr<AutoNode>& cur,
                          std::unordered_set<AutoNode*>& visited,
                          std::vector<std::shared_ptr<AutoNode>>& topo) {
  if (!cur || visited.count(cur.get())) return;
  visited.insert(cur.get());
  for (const auto& p : cur->parents) {
    build_topo(p, visited, topo);
  }
  topo.push_back(cur);
}

void Variable::backward() {
  if (node->data.total_size != 1) {
    throw std::runtime_error("backward currently expects scalar output");
  }

  std::unordered_set<AutoNode*> visited;
  std::vector<std::shared_ptr<AutoNode>> topo;
  build_topo(node, visited, topo);

  for (auto& n : topo) {
    n->grad = Tensor::ones(n->data.get_dims(), n->data.get_device_type(), n->data.get_device_index()).mult(0.0);
  }
  node->grad = Tensor::ones({1}, node->data.get_device_type(), node->data.get_device_index());

  for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
    if ((*it)->backward_fn) {
      (*it)->backward_fn();
    }
  }
}
