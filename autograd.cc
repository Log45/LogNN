#include "autograd.h"

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
