#include "autograd.h"

#include <stdexcept>

namespace {
void accumulate_grad(std::shared_ptr<AutoNode> n, const Tensor& g) {
  if (!n->requires_grad) return;
  n->grad = n->grad.add(g);
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
