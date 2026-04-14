#pragma once

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "tensor.h"

struct MaxPool2dFwdState {
  std::vector<size_t> argmax_cpu;
  std::shared_ptr<void> argmax_cuda;
  std::shared_ptr<void> argmax_mlx_u32;
  size_t argmax_len = 0;
};

inline size_t conv_output_length(size_t in_len, size_t kernel, size_t stride, size_t pad_per_side) {
  const size_t padded = in_len + 2 * pad_per_side;
  if (padded < kernel) {
    throw std::runtime_error("conv_output_length: padded spatial size < kernel");
  }
  return (padded - kernel) / stride + 1;  // integer division floors for size_t
}

std::vector<double> nchw_pad(const std::vector<double>& x, size_t N, size_t C, size_t H, size_t W,
                             size_t ph, size_t pw);

void nchw_unpad_grad(const std::vector<double>& dxp, size_t N, size_t C, size_t H, size_t W, size_t ph,
                     size_t pw, std::vector<double>& dx);

void conv_im2col(const std::vector<double>& xp, size_t N, size_t Ci, size_t Hp, size_t Wp, size_t kH,
                 size_t kW, size_t sh, size_t sw, size_t Ho, size_t Wo, std::vector<double>& col);

void conv_col2im(const std::vector<double>& col, size_t N, size_t Ci, size_t Hp, size_t Wp, size_t kH,
                 size_t kW, size_t sh, size_t sw, size_t Ho, size_t Wo, std::vector<double>& dxp);

Tensor conv2d_forward(const Tensor& x, const Tensor& w, size_t sh, size_t sw, size_t ph, size_t pw,
                      size_t& Ho, size_t& Wo);

std::pair<Tensor, Tensor> conv2d_backward(const Tensor& grad_y, const Tensor& x, const Tensor& w, size_t sh,
                                            size_t sw, size_t ph, size_t pw, size_t Ho, size_t Wo);

void maxpool2d_forward(const std::vector<double>& x, size_t N, size_t C, size_t H, size_t W, size_t kH,
                       size_t kW, size_t sh, size_t sw, size_t ph, size_t pw, std::vector<double>& y,
                       std::vector<size_t>& argmax, size_t& Ho, size_t& Wo);

void maxpool2d_backward(const std::vector<double>& dy, const std::vector<size_t>& argmax, size_t N,
                        size_t C, size_t H, size_t W, size_t ph, size_t pw, size_t Ho, size_t Wo,
                        std::vector<double>& dx);

void avgpool2d_forward(const std::vector<double>& x, size_t N, size_t C, size_t H, size_t W, size_t kH,
                       size_t kW, size_t sh, size_t sw, size_t ph, size_t pw, std::vector<double>& y,
                       size_t& Ho, size_t& Wo);

void avgpool2d_backward(const std::vector<double>& dy, size_t N, size_t C, size_t H, size_t W, size_t kH,
                        size_t kW, size_t sh, size_t sw, size_t ph, size_t pw, size_t Ho, size_t Wo,
                        std::vector<double>& dx);

Tensor conv_transpose2d_forward(const Tensor& x, const Tensor& w, size_t sh, size_t sw, size_t ph, size_t pw,
                                size_t out_ph, size_t out_pw, size_t& Ho, size_t& Wo);

void conv_transpose2d_backward(const std::vector<double>& dy, const std::vector<double>& x,
                               const std::vector<double>& w_flat, size_t N, size_t Ci, size_t Hi, size_t Wi,
                               size_t Co, size_t kH, size_t kW, size_t sh, size_t sw, size_t ph, size_t pw,
                               size_t out_ph, size_t out_pw, size_t Ho, size_t Wo,
                               std::vector<double>& dx, std::vector<double>& dw);

std::pair<Tensor, Tensor> conv_transpose2d_backward(const Tensor& grad_y, const Tensor& x, const Tensor& w,
                                                     size_t sh, size_t sw, size_t ph, size_t pw, size_t out_ph,
                                                     size_t out_pw, size_t Ho, size_t Wo);

Tensor maxpool2d_forward_tensor(const Tensor& x, size_t kH, size_t kW, size_t sh, size_t sw, size_t ph, size_t pw,
                                MaxPool2dFwdState& st, size_t& Ho, size_t& Wo);

Tensor maxpool2d_backward_tensor(const Tensor& dy, const MaxPool2dFwdState& st, size_t N, size_t C, size_t H,
                                 size_t W, size_t kH, size_t kW, size_t sh, size_t sw, size_t ph, size_t pw,
                                 size_t Ho, size_t Wo);

Tensor avgpool2d_forward_tensor(const Tensor& x, size_t kH, size_t kW, size_t sh, size_t sw, size_t ph, size_t pw,
                                size_t& Ho, size_t& Wo);

Tensor avgpool2d_backward_tensor(const Tensor& dy, size_t N, size_t C, size_t H, size_t W, size_t kH, size_t kW,
                                 size_t sh, size_t sw, size_t ph, size_t pw, size_t Ho, size_t Wo);
