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
  static Variable reshape(const Variable& x, const std::vector<size_t>& new_dims);
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

  /// weight [V, D], token_ids [T] (values 0..V-1 as doubles). Returns [T, D].
  static Variable embedding_gather(const Variable& weight, const Tensor& token_ids);
  /// logits [T, V], tokens [T] — next-token LM loss: mean CE(logits[i], tokens[i+1]) for i=0..T-2.
  static Variable cross_entropy_next_token_lm(const Variable& logits, const Tensor& tokens_T);

  /// y [N,C,H,W], bias [1,C,1,1].
  static Variable add_nchw_bias(const Variable& y, const Variable& bias);

  /// NCHW conv2d. weight [C_out, C_in, kH, kW]; optional bias [1, C_out, 1, 1].
  static Variable conv2d(const Variable& x, const Variable& weight, const Variable& bias, size_t stride_h,
                         size_t stride_w, size_t pad_h, size_t pad_w, bool use_bias);

  static Variable max_pool2d(const Variable& x, size_t kernel_h, size_t kernel_w, size_t stride_h,
                             size_t stride_w, size_t pad_h, size_t pad_w);

  static Variable avg_pool2d(const Variable& x, size_t kernel_h, size_t kernel_w, size_t stride_h,
                             size_t stride_w, size_t pad_h, size_t pad_w);

  /// weight [C_in, C_out, kH, kW] (PyTorch ConvTranspose2d layout). output_padding adds to output H/W.
  static Variable conv_transpose2d(const Variable& x, const Variable& weight, size_t stride_h, size_t stride_w,
                                     size_t pad_h, size_t pad_w, size_t output_pad_h, size_t output_pad_w);

  /// logits [N, C]; same stability as log(softmax) along class dim.
  static Variable log_softmax_last_dim(const Variable& x);

  /// logits [N, num_classes], target [N] with class indices as doubles. ignore_index skipped in mean (default -100).
  static Variable cross_entropy_logits(const Variable& logits, const Tensor& target, double ignore_index = -100.0);

  /// NCHW batch norm. gamma, beta [1, C, 1, 1]; running_mean / running_var updated in-place when training.
  static Variable batch_norm2d(const Variable& x, const Variable& gamma, const Variable& beta, Tensor& running_mean,
                               Tensor& running_var, double momentum, double eps, bool training);

  void backward();

 private:
  static void build_topo(const std::shared_ptr<AutoNode>& cur,
                         std::unordered_set<AutoNode*>& visited,
                         std::vector<std::shared_ptr<AutoNode>>& topo);
};
