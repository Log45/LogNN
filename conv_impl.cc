#include "conv_impl.h"

#include <algorithm>
#include <cmath>
#include <cstdint>

#if defined(WITH_CUDA) || defined(WITH_MLX)
#include "tensor_kernels.h"
#endif

namespace {

inline size_t nchw_index(size_t n, size_t c, size_t h, size_t w, size_t C, size_t H, size_t W) {
  return ((n * C + c) * H + h) * W + w;
}

inline size_t conv_out_dim_floor(size_t in_len, size_t kernel, size_t stride, size_t pad_per_side) {
  const int64_t padded = static_cast<int64_t>(in_len) + 2 * static_cast<int64_t>(pad_per_side);
  const int64_t k = static_cast<int64_t>(kernel);
  const int64_t s = static_cast<int64_t>(stride);
  if (padded < k) {
    throw std::runtime_error("conv: padded spatial size smaller than kernel");
  }
  return static_cast<size_t>((padded - k) / s + 1);
}

}  // namespace

std::vector<double> nchw_pad(const std::vector<double>& x, size_t N, size_t C, size_t H, size_t W,
                             size_t ph, size_t pw) {
  const size_t Hp = H + 2 * ph;
  const size_t Wp = W + 2 * pw;
  std::vector<double> xp(N * C * Hp * Wp, 0.0);
  for (size_t n = 0; n < N; ++n) {
    for (size_t c = 0; c < C; ++c) {
      for (size_t h = 0; h < H; ++h) {
        for (size_t w = 0; w < W; ++w) {
          const double v = x[nchw_index(n, c, h, w, C, H, W)];
          xp[nchw_index(n, c, h + ph, w + pw, C, Hp, Wp)] = v;
        }
      }
    }
  }
  return xp;
}

void nchw_unpad_grad(const std::vector<double>& dxp, size_t N, size_t C, size_t H, size_t W, size_t ph,
                     size_t pw, std::vector<double>& dx) {
  const size_t Hp = H + 2 * ph;
  const size_t Wp = W + 2 * pw;
  dx.resize(N * C * H * W);
  for (size_t n = 0; n < N; ++n) {
    for (size_t c = 0; c < C; ++c) {
      for (size_t h = 0; h < H; ++h) {
        for (size_t w = 0; w < W; ++w) {
          dx[nchw_index(n, c, h, w, C, H, W)] = dxp[nchw_index(n, c, h + ph, w + pw, C, Hp, Wp)];
        }
      }
    }
  }
}

void conv_im2col(const std::vector<double>& xp, size_t N, size_t Ci, size_t Hp, size_t Wp, size_t kH,
                 size_t kW, size_t sh, size_t sw, size_t Ho, size_t Wo, std::vector<double>& col) {
  const size_t K = Ci * kH * kW;
  const size_t M = N * Ho * Wo;
  col.assign(M * K, 0.0);
  size_t m = 0;
  for (size_t n = 0; n < N; ++n) {
    for (size_t oh = 0; oh < Ho; ++oh) {
      for (size_t ow = 0; ow < Wo; ++ow) {
        size_t kk = 0;
        for (size_t ci = 0; ci < Ci; ++ci) {
          for (size_t kh = 0; kh < kH; ++kh) {
            for (size_t kw = 0; kw < kW; ++kw) {
              const size_t ih = oh * sh + kh;
              const size_t iw = ow * sw + kw;
              col[m * K + kk] = xp[nchw_index(n, ci, ih, iw, Ci, Hp, Wp)];
              ++kk;
            }
          }
        }
        ++m;
      }
    }
  }
}

void conv_col2im(const std::vector<double>& col, size_t N, size_t Ci, size_t Hp, size_t Wp, size_t kH,
                 size_t kW, size_t sh, size_t sw, size_t Ho, size_t Wo, std::vector<double>& dxp) {
  const size_t K = Ci * kH * kW;
  dxp.assign(N * Ci * Hp * Wp, 0.0);
  size_t m = 0;
  for (size_t n = 0; n < N; ++n) {
    for (size_t oh = 0; oh < Ho; ++oh) {
      for (size_t ow = 0; ow < Wo; ++ow) {
        size_t kk = 0;
        for (size_t ci = 0; ci < Ci; ++ci) {
          for (size_t kh = 0; kh < kH; ++kh) {
            for (size_t kw = 0; kw < kW; ++kw) {
              const size_t ih = oh * sh + kh;
              const size_t iw = ow * sw + kw;
              dxp[nchw_index(n, ci, ih, iw, Ci, Hp, Wp)] += col[m * K + kk];
              ++kk;
            }
          }
        }
        ++m;
      }
    }
  }
}

static Tensor conv2d_forward_cpu(const Tensor& x, const Tensor& w, size_t sh, size_t sw, size_t ph, size_t pw,
                                 size_t Ho, size_t Wo) {
  const auto xd = x.get_dims();
  const auto wd = w.get_dims();
  const size_t N = xd[0], Ci = xd[1], H = xd[2], W = xd[3];
  const size_t Co = wd[0], kH = wd[2], kW = wd[3];
  const size_t K = Ci * kH * kW;
  const size_t M = N * Ho * Wo;
  const size_t Hp = H + 2 * ph;
  const size_t Wp = W + 2 * pw;
  std::vector<double> xh = x.get_data();
  std::vector<double> xp = nchw_pad(xh, N, Ci, H, W, ph, pw);
  std::vector<double> col;
  conv_im2col(xp, N, Ci, Hp, Wp, kH, kW, sh, sw, Ho, Wo, col);
  std::vector<double> wh = w.get_data();
  std::vector<double> y_flat(N * Co * Ho * Wo, 0.0);
  for (size_t m = 0; m < M; ++m) {
    for (size_t co = 0; co < Co; ++co) {
      double acc = 0.0;
      for (size_t k = 0; k < K; ++k) {
        acc += col[m * K + k] * wh[co * K + k];
      }
      const size_t n = m / (Ho * Wo);
      const size_t rem = m % (Ho * Wo);
      const size_t oh = rem / Wo;
      const size_t ow = rem % Wo;
      y_flat[nchw_index(n, co, oh, ow, Co, Ho, Wo)] = acc;
    }
  }
  return Tensor::from_data({N, Co, Ho, Wo}, y_flat, x.get_device_type(), x.get_device_index());
}

Tensor conv2d_forward(const Tensor& x, const Tensor& w, size_t sh, size_t sw, size_t ph, size_t pw,
                      size_t& Ho, size_t& Wo) {
  const auto xd = x.get_dims();
  const auto wd = w.get_dims();
  if (xd.size() != 4 || wd.size() != 4) {
    throw std::runtime_error("conv2d_forward: x and w must be 4D NCHW");
  }
  const size_t Ci = xd[1], H = xd[2], W = xd[3];
  const size_t Ci_w = wd[1], kH = wd[2], kW = wd[3];
  if (Ci_w != Ci) {
    throw std::runtime_error("conv2d_forward: input channels != weight in_channels");
  }
  Ho = conv_out_dim_floor(H, kH, sh, ph);
  Wo = conv_out_dim_floor(W, kW, sw, pw);
#if defined(WITH_CUDA)
  if (x.device.type == DeviceType::CUDA) {
    Tensor::ensure_same_device(x, w, "conv2d_forward");
    const size_t N = xd[0], Co = wd[0];
    const size_t out_sz = N * Co * Ho * Wo;
    double* yptr = backend_alloc(x.device, out_sz);
    gpu_conv2d_forward_nchw(x.data, w.data, yptr, N, Ci, H, W, Co, kH, kW, sh, sw, ph, pw, Ho, Wo);
    return Tensor({N, Co, Ho, Wo}, yptr, out_sz, x.device);
  }
#endif
#if defined(WITH_MLX)
  if (x.device.type == DeviceType::MLX) {
    Tensor::ensure_same_device(x, w, "conv2d_forward");
    const size_t N = xd[0], Co = wd[0];
    const size_t out_sz = N * Co * Ho * Wo;
    double* yptr = backend_alloc(x.device, out_sz);
    mlx_conv2d_forward_nchw(x.data, w.data, yptr, N, Ci, H, W, Co, kH, kW, sh, sw, ph, pw, Ho, Wo);
    return Tensor({N, Co, Ho, Wo}, yptr, out_sz, x.device);
  }
#endif
  return conv2d_forward_cpu(x, w, sh, sw, ph, pw, Ho, Wo);
}


static std::pair<Tensor, Tensor> conv2d_backward_cpu(const Tensor& grad_y, const Tensor& x, const Tensor& w,
                                                     size_t sh, size_t sw, size_t ph, size_t pw, size_t Ho,
                                                     size_t Wo) {
  const auto xd = x.get_dims();
  const auto wd = w.get_dims();
  const size_t N = xd[0], Ci = xd[1], H = xd[2], W = xd[3];
  const size_t Co = wd[0], Ci_w = wd[1], kH = wd[2], kW = wd[3];
  (void)Ci_w;
  const size_t Hp = H + 2 * ph;
  const size_t Wp = W + 2 * pw;
  const size_t K = Ci * kH * kW;
  const size_t M = N * Ho * Wo;
  std::vector<double> xh = x.get_data();
  std::vector<double> xp = nchw_pad(xh, N, Ci, H, W, ph, pw);
  std::vector<double> col;
  conv_im2col(xp, N, Ci, Hp, Wp, kH, kW, sh, sw, Ho, Wo, col);
  std::vector<double> dyh = grad_y.get_data();
  std::vector<double> wh = w.get_data();
  std::vector<double> dcol(M * K, 0.0);
  for (size_t m = 0; m < M; ++m) {
    const size_t n = m / (Ho * Wo);
    const size_t rem = m % (Ho * Wo);
    const size_t oh = rem / Wo;
    const size_t ow = rem % Wo;
    for (size_t k = 0; k < K; ++k) {
      double s = 0.0;
      for (size_t co = 0; co < Co; ++co) {
        const double dyv = dyh[nchw_index(n, co, oh, ow, Co, Ho, Wo)];
        s += dyv * wh[co * K + k];
      }
      dcol[m * K + k] = s;
    }
  }
  std::vector<double> dwh(Co * K, 0.0);
  for (size_t co = 0; co < Co; ++co) {
    for (size_t k = 0; k < K; ++k) {
      double s = 0.0;
      for (size_t m = 0; m < M; ++m) {
        const size_t n = m / (Ho * Wo);
        const size_t rem = m % (Ho * Wo);
        const size_t oh = rem / Wo;
        const size_t ow = rem % Wo;
        s += dyh[nchw_index(n, co, oh, ow, Co, Ho, Wo)] * col[m * K + k];
      }
      dwh[co * K + k] = s;
    }
  }
  std::vector<double> dxp;
  conv_col2im(dcol, N, Ci, Hp, Wp, kH, kW, sh, sw, Ho, Wo, dxp);
  std::vector<double> dxh;
  nchw_unpad_grad(dxp, N, Ci, H, W, ph, pw, dxh);
  Tensor grad_x = Tensor::from_data({N, Ci, H, W}, dxh, x.get_device_type(), x.get_device_index());
  Tensor grad_w = Tensor::from_data({Co, Ci, kH, kW}, dwh, w.get_device_type(), w.get_device_index());
  return {std::move(grad_x), std::move(grad_w)};
}

std::pair<Tensor, Tensor> conv2d_backward(const Tensor& grad_y, const Tensor& x, const Tensor& w, size_t sh,
                                        size_t sw, size_t ph, size_t pw, size_t Ho, size_t Wo) {
  const auto xd = x.get_dims();
  const auto wd = w.get_dims();
  if (wd[1] != xd[1]) {
    throw std::runtime_error("conv2d_backward: channel mismatch");
  }
#if defined(WITH_CUDA)
  if (x.device.type == DeviceType::CUDA) {
    const size_t N = xd[0], Ci = xd[1], H = xd[2], W = xd[3];
    const size_t Co = wd[0], kH = wd[2], kW = wd[3];
    Tensor::ensure_same_device(grad_y, x, "conv2d_backward");
    Tensor::ensure_same_device(x, w, "conv2d_backward");
    double* dxp = backend_alloc(x.device, N * Ci * H * W);
    double* dwp = backend_alloc(w.device, Co * Ci * kH * kW);
    gpu_conv2d_backward_nchw(grad_y.data, x.data, w.data, dxp, dwp, N, Ci, H, W, Co, kH, kW, sh, sw, ph, pw, Ho, Wo);
    Tensor gx({N, Ci, H, W}, dxp, N * Ci * H * W, x.device);
    Tensor gw({Co, Ci, kH, kW}, dwp, Co * Ci * kH * kW, w.device);
    return {std::move(gx), std::move(gw)};
  }
#endif
#if defined(WITH_MLX)
  if (x.device.type == DeviceType::MLX) {
    const size_t N = xd[0], Ci = xd[1], H = xd[2], W = xd[3];
    const size_t Co = wd[0], kH = wd[2], kW = wd[3];
    Tensor::ensure_same_device(grad_y, x, "conv2d_backward");
    Tensor::ensure_same_device(x, w, "conv2d_backward");
    double* dxp = backend_alloc(x.device, N * Ci * H * W);
    double* dwp = backend_alloc(w.device, Co * Ci * kH * kW);
    mlx_conv2d_backward_nchw(grad_y.data, x.data, w.data, dxp, dwp, N, Ci, H, W, Co, kH, kW, sh, sw, ph, pw, Ho, Wo);
    Tensor gx({N, Ci, H, W}, dxp, N * Ci * H * W, x.device);
    Tensor gw({Co, Ci, kH, kW}, dwp, Co * Ci * kH * kW, w.device);
    return {std::move(gx), std::move(gw)};
  }
#endif
  return conv2d_backward_cpu(grad_y, x, w, sh, sw, ph, pw, Ho, Wo);
}

void maxpool2d_forward(const std::vector<double>& x, size_t N, size_t C, size_t H, size_t W, size_t kH,
                       size_t kW, size_t sh, size_t sw, size_t ph, size_t pw, std::vector<double>& y,
                       std::vector<size_t>& argmax, size_t& Ho, size_t& Wo) {
  const size_t Hp = H + 2 * ph;
  const size_t Wp = W + 2 * pw;
  Ho = conv_out_dim_floor(H, kH, sh, ph);
  Wo = conv_out_dim_floor(W, kW, sw, pw);
  std::vector<double> xp = nchw_pad(x, N, C, H, W, ph, pw);
  y.resize(N * C * Ho * Wo);
  argmax.resize(N * C * Ho * Wo);
  for (size_t n = 0; n < N; ++n) {
    for (size_t c = 0; c < C; ++c) {
      for (size_t oh = 0; oh < Ho; ++oh) {
        for (size_t ow = 0; ow < Wo; ++ow) {
          double best = 0.0;
          size_t best_idx = 0;
          bool first = true;
          for (size_t kh = 0; kh < kH; ++kh) {
            for (size_t kw = 0; kw < kW; ++kw) {
              const size_t ih = oh * sh + kh;
              const size_t iw = ow * sw + kw;
              const size_t li = nchw_index(n, c, ih, iw, C, Hp, Wp);
              const double v = xp[li];
              if (first || v > best) {
                best = v;
                best_idx = li;
                first = false;
              }
            }
          }
          const size_t yo = nchw_index(n, c, oh, ow, C, Ho, Wo);
          y[yo] = best;
          argmax[yo] = best_idx;
        }
      }
    }
  }
}

void maxpool2d_backward(const std::vector<double>& dy, const std::vector<size_t>& argmax, size_t N,
                        size_t C, size_t H, size_t W, size_t ph, size_t pw, size_t Ho, size_t Wo,
                        std::vector<double>& dx) {
  const size_t Hp = H + 2 * ph;
  const size_t Wp = W + 2 * pw;
  std::vector<double> dxp(N * C * Hp * Wp, 0.0);
  for (size_t n = 0; n < N; ++n) {
    for (size_t c = 0; c < C; ++c) {
      for (size_t oh = 0; oh < Ho; ++oh) {
        for (size_t ow = 0; ow < Wo; ++ow) {
          const size_t yi = nchw_index(n, c, oh, ow, C, Ho, Wo);
          dxp[argmax[yi]] += dy[yi];
        }
      }
    }
  }
  nchw_unpad_grad(dxp, N, C, H, W, ph, pw, dx);
}

void avgpool2d_forward(const std::vector<double>& x, size_t N, size_t C, size_t H, size_t W, size_t kH,
                       size_t kW, size_t sh, size_t sw, size_t ph, size_t pw, std::vector<double>& y,
                       size_t& Ho, size_t& Wo) {
  const size_t Hp = H + 2 * ph;
  const size_t Wp = W + 2 * pw;
  Ho = conv_out_dim_floor(H, kH, sh, ph);
  Wo = conv_out_dim_floor(W, kW, sw, pw);
  std::vector<double> xp = nchw_pad(x, N, C, H, W, ph, pw);
  y.resize(N * C * Ho * Wo);
  const double scale = 1.0 / static_cast<double>(kH * kW);
  for (size_t n = 0; n < N; ++n) {
    for (size_t c = 0; c < C; ++c) {
      for (size_t oh = 0; oh < Ho; ++oh) {
        for (size_t ow = 0; ow < Wo; ++ow) {
          double s = 0.0;
          for (size_t kh = 0; kh < kH; ++kh) {
            for (size_t kw = 0; kw < kW; ++kw) {
              const size_t ih = oh * sh + kh;
              const size_t iw = ow * sw + kw;
              s += xp[nchw_index(n, c, ih, iw, C, Hp, Wp)];
            }
          }
          y[nchw_index(n, c, oh, ow, C, Ho, Wo)] = s * scale;
        }
      }
    }
  }
}

void avgpool2d_backward(const std::vector<double>& dy, size_t N, size_t C, size_t H, size_t W, size_t kH,
                        size_t kW, size_t sh, size_t sw, size_t ph, size_t pw, size_t Ho, size_t Wo,
                        std::vector<double>& dx) {
  const size_t Hp = H + 2 * ph;
  const size_t Wp = W + 2 * pw;
  std::vector<double> dxp(N * C * Hp * Wp, 0.0);
  const double scale = 1.0 / static_cast<double>(kH * kW);
  for (size_t n = 0; n < N; ++n) {
    for (size_t c = 0; c < C; ++c) {
      for (size_t oh = 0; oh < Ho; ++oh) {
        for (size_t ow = 0; ow < Wo; ++ow) {
          const double g = dy[nchw_index(n, c, oh, ow, C, Ho, Wo)] * scale;
          for (size_t kh = 0; kh < kH; ++kh) {
            for (size_t kw = 0; kw < kW; ++kw) {
              const size_t ih = oh * sh + kh;
              const size_t iw = ow * sw + kw;
              dxp[nchw_index(n, c, ih, iw, C, Hp, Wp)] += g;
            }
          }
        }
      }
    }
  }
  nchw_unpad_grad(dxp, N, C, H, W, ph, pw, dx);
}

Tensor conv_transpose2d_forward(const Tensor& x, const Tensor& w, size_t sh, size_t sw, size_t ph, size_t pw,
                                size_t out_ph, size_t out_pw, size_t& Ho, size_t& Wo) {
  const auto xd = x.get_dims();
  const auto wd = w.get_dims();
  if (xd.size() != 4 || wd.size() != 4) {
    throw std::runtime_error("conv_transpose2d_forward: 4D tensors required");
  }
  const size_t N = xd[0], Ci = xd[1], Hi = xd[2], Wi = xd[3];
  const size_t Ci_w = wd[0], Co = wd[1], kH = wd[2], kW = wd[3];
  if (Ci_w != Ci) {
    throw std::runtime_error("conv_transpose2d: weight in_channels mismatch");
  }
  const int64_t hin = static_cast<int64_t>(Hi);
  const int64_t win = static_cast<int64_t>(Wi);
  const int64_t kh = static_cast<int64_t>(kH);
  const int64_t kw = static_cast<int64_t>(kW);
  const int64_t s_h = static_cast<int64_t>(sh);
  const int64_t s_w = static_cast<int64_t>(sw);
  const int64_t p_h = static_cast<int64_t>(ph);
  const int64_t p_w = static_cast<int64_t>(pw);
  const int64_t op_h = static_cast<int64_t>(out_ph);
  const int64_t op_w = static_cast<int64_t>(out_pw);
  Ho = static_cast<size_t>((hin - 1) * s_h - 2 * p_h + kh + op_h);
  Wo = static_cast<size_t>((win - 1) * s_w - 2 * p_w + kw + op_w);
  if (Ho <= 0 || Wo <= 0) {
    throw std::runtime_error("conv_transpose2d_forward: non-positive output size");
  }
#if defined(WITH_CUDA)
  if (x.device.type == DeviceType::CUDA) {
    Tensor::ensure_same_device(x, w, "conv_transpose2d_forward");
    const size_t out_sz = N * Co * Ho * Wo;
    double* yptr = backend_alloc(x.device, out_sz);
    gpu_conv_transpose2d_forward_nchw(x.data, w.data, yptr, N, Ci, Hi, Wi, Co, kH, kW, sh, sw, ph, pw, Ho, Wo);
    return Tensor({N, Co, Ho, Wo}, yptr, out_sz, x.device);
  }
#endif
#if defined(WITH_MLX)
  if (x.device.type == DeviceType::MLX) {
    Tensor::ensure_same_device(x, w, "conv_transpose2d_forward");
    const size_t out_sz = N * Co * Ho * Wo;
    double* yptr = backend_alloc(x.device, out_sz);
    mlx_conv_transpose2d_forward_nchw(x.data, w.data, yptr, N, Ci, Hi, Wi, Co, kH, kW, sh, sw, ph, pw, Ho, Wo);
    return Tensor({N, Co, Ho, Wo}, yptr, out_sz, x.device);
  }
#endif
  std::vector<double> xh = x.get_data();
  std::vector<double> wh = w.get_data();
  std::vector<double> yh(N * Co * Ho * Wo, 0.0);
  for (size_t n = 0; n < N; ++n) {
    for (size_t ci = 0; ci < Ci; ++ci) {
      for (size_t hi = 0; hi < Hi; ++hi) {
        for (size_t wi = 0; wi < Wi; ++wi) {
          const double xv = xh[nchw_index(n, ci, hi, wi, Ci, Hi, Wi)];
          for (size_t co = 0; co < Co; ++co) {
            for (size_t kh_i = 0; kh_i < kH; ++kh_i) {
              for (size_t kw_i = 0; kw_i < kW; ++kw_i) {
                const int64_t ho = static_cast<int64_t>(hi) * s_h + static_cast<int64_t>(kh_i) - p_h;
                const int64_t wo = static_cast<int64_t>(wi) * s_w + static_cast<int64_t>(kw_i) - p_w;
                if (ho < 0 || wo < 0 || ho >= static_cast<int64_t>(Ho) || wo >= static_cast<int64_t>(Wo)) {
                  continue;
                }
                const size_t wi_idx = ((ci * Co + co) * kH + kh_i) * kW + kw_i;
                yh[nchw_index(n, co, static_cast<size_t>(ho), static_cast<size_t>(wo), Co, Ho, Wo)] +=
                    xv * wh[wi_idx];
              }
            }
          }
        }
      }
    }
  }
  (void)op_h;
  (void)op_w;
  return Tensor::from_data({N, Co, Ho, Wo}, yh, x.get_device_type(), x.get_device_index());
}

void conv_transpose2d_backward(const std::vector<double>& dy, const std::vector<double>& x,
                               const std::vector<double>& w_flat, size_t N, size_t Ci, size_t Hi, size_t Wi,
                               size_t Co, size_t kH, size_t kW, size_t sh, size_t sw, size_t ph, size_t pw,
                               size_t /*out_ph*/, size_t /*out_pw*/, size_t Ho, size_t Wo, std::vector<double>& dx,
                               std::vector<double>& dw) {
  dx.assign(N * Ci * Hi * Wi, 0.0);
  dw.assign(Ci * Co * kH * kW, 0.0);
  const int64_t s_h = static_cast<int64_t>(sh);
  const int64_t s_w = static_cast<int64_t>(sw);
  const int64_t p_h = static_cast<int64_t>(ph);
  const int64_t p_w = static_cast<int64_t>(pw);
  for (size_t n = 0; n < N; ++n) {
    for (size_t ci = 0; ci < Ci; ++ci) {
      for (size_t hi = 0; hi < Hi; ++hi) {
        for (size_t wi = 0; wi < Wi; ++wi) {
          const size_t xi = nchw_index(n, ci, hi, wi, Ci, Hi, Wi);
          for (size_t co = 0; co < Co; ++co) {
            for (size_t kh_i = 0; kh_i < kH; ++kh_i) {
              for (size_t kw_i = 0; kw_i < kW; ++kw_i) {
                const int64_t ho = static_cast<int64_t>(hi) * s_h + static_cast<int64_t>(kh_i) - p_h;
                const int64_t wo = static_cast<int64_t>(wi) * s_w + static_cast<int64_t>(kw_i) - p_w;
                if (ho < 0 || wo < 0 || ho >= static_cast<int64_t>(Ho) || wo >= static_cast<int64_t>(Wo)) {
                  continue;
                }
                const size_t yi =
                    nchw_index(n, co, static_cast<size_t>(ho), static_cast<size_t>(wo), Co, Ho, Wo);
                const double g = dy[yi];
                if (g == 0.0) continue;
                const size_t wi_idx = ((ci * Co + co) * kH + kh_i) * kW + kw_i;
                dx[xi] += g * w_flat[wi_idx];
                dw[wi_idx] += g * x[xi];
              }
            }
          }
        }
      }
    }
  }
}

std::pair<Tensor, Tensor> conv_transpose2d_backward(const Tensor& grad_y, const Tensor& x, const Tensor& w,
                                                    size_t sh, size_t sw, size_t ph, size_t pw, size_t /*out_ph*/,
                                                    size_t /*out_pw*/, size_t Ho, size_t Wo) {
  const auto xd = x.get_dims();
  const auto wd = w.get_dims();
  const auto gyd = grad_y.get_dims();
  const size_t N = xd[0], Ci = xd[1], Hi = xd[2], Wi = xd[3];
  const size_t Ci_w = wd[0], Co = wd[1], kH = wd[2], kW = wd[3];
  if (Ci_w != Ci || gyd != std::vector<size_t>({N, Co, Ho, Wo})) {
    throw std::runtime_error("conv_transpose2d_backward: shape mismatch");
  }
#if defined(WITH_CUDA)
  if (x.device.type == DeviceType::CUDA) {
    Tensor::ensure_same_device(grad_y, x, "conv_transpose2d_backward");
    Tensor::ensure_same_device(x, w, "conv_transpose2d_backward");
    double* dxp = backend_alloc(x.device, N * Ci * Hi * Wi);
    double* dwp = backend_alloc(w.device, Ci * Co * kH * kW);
    gpu_conv_transpose2d_backward_nchw(grad_y.data, x.data, w.data, dxp, dwp, N, Ci, Hi, Wi, Co, kH, kW, sh, sw, ph,
                                       pw, Ho, Wo);
    Tensor gx({N, Ci, Hi, Wi}, dxp, N * Ci * Hi * Wi, x.device);
    Tensor gw({Ci, Co, kH, kW}, dwp, Ci * Co * kH * kW, w.device);
    return {std::move(gx), std::move(gw)};
  }
#endif
#if defined(WITH_MLX)
  if (x.device.type == DeviceType::MLX) {
    Tensor::ensure_same_device(grad_y, x, "conv_transpose2d_backward");
    Tensor::ensure_same_device(x, w, "conv_transpose2d_backward");
    double* dxp = backend_alloc(x.device, N * Ci * Hi * Wi);
    double* dwp = backend_alloc(w.device, Ci * Co * kH * kW);
    mlx_conv_transpose2d_backward_nchw(grad_y.data, x.data, w.data, dxp, dwp, N, Ci, Hi, Wi, Co, kH, kW, sh, sw, ph,
                                       pw, Ho, Wo);
    Tensor gx({N, Ci, Hi, Wi}, dxp, N * Ci * Hi * Wi, x.device);
    Tensor gw({Ci, Co, kH, kW}, dwp, Ci * Co * kH * kW, w.device);
    return {std::move(gx), std::move(gw)};
  }
#endif
  std::vector<double> dyh = grad_y.get_data();
  std::vector<double> xh = x.get_data();
  std::vector<double> wh = w.get_data();
  std::vector<double> dx;
  std::vector<double> dw;
  conv_transpose2d_backward(dyh, xh, wh, N, Ci, Hi, Wi, Co, kH, kW, sh, sw, ph, pw, 0, 0, Ho, Wo, dx, dw);
  Tensor gx = Tensor::from_data({N, Ci, Hi, Wi}, dx, x.get_device_type(), x.get_device_index());
  Tensor gw = Tensor::from_data({Ci, Co, kH, kW}, dw, w.get_device_type(), w.get_device_index());
  return {std::move(gx), std::move(gw)};
}

Tensor maxpool2d_forward_tensor(const Tensor& x, size_t kH, size_t kW, size_t sh, size_t sw, size_t ph, size_t pw,
                                MaxPool2dFwdState& st, size_t& Ho, size_t& Wo) {
  const auto xd = x.get_dims();
  if (xd.size() != 4) {
    throw std::runtime_error("maxpool2d_forward_tensor: NCHW required");
  }
  const size_t N = xd[0], C = xd[1], H = xd[2], W = xd[3];
  Ho = conv_out_dim_floor(H, kH, sh, ph);
  Wo = conv_out_dim_floor(W, kW, sw, pw);
#if defined(WITH_CUDA)
  if (x.device.type == DeviceType::CUDA) {
    const size_t Hp = H + 2 * ph;
    const size_t Wp = W + 2 * pw;
    const size_t nout = N * C * Ho * Wo;
    st.argmax_cpu.clear();
    st.argmax_mlx_u32.reset();
    st.argmax_len = nout;
    unsigned long long* aptr = gpu_alloc_ull(nout);
    st.argmax_cuda = std::shared_ptr<void>(aptr, [](void* p) {
#if defined(WITH_CUDA)
      if (p) {
        gpu_free_ull(static_cast<unsigned long long*>(p));
      }
#endif
    });
    double* yptr = backend_alloc(x.device, nout);
    gpu_maxpool2d_forward_nchw(x.data, yptr, aptr, N, C, H, W, kH, kW, sh, sw, ph, pw, Ho, Wo, Hp, Wp);
    return Tensor({N, C, Ho, Wo}, yptr, nout, x.device);
  }
#endif
#if defined(WITH_MLX)
  if (x.device.type == DeviceType::MLX) {
    const size_t Hp = H + 2 * ph;
    const size_t Wp = W + 2 * pw;
    const size_t nout = N * C * Ho * Wo;
    st.argmax_cpu.clear();
    st.argmax_cuda.reset();
    st.argmax_len = nout;
    uint32_t* aptr = mlx_alloc_u32(nout);
    st.argmax_mlx_u32 = std::shared_ptr<void>(aptr, [](void* p) {
#if defined(WITH_MLX)
      if (p) {
        mlx_free_u32(static_cast<uint32_t*>(p));
      }
#endif
    });
    double* yptr = backend_alloc(x.device, nout);
    mlx_maxpool2d_forward_nchw(x.data, yptr, aptr, N, C, H, W, kH, kW, sh, sw, ph, pw, Ho, Wo, Hp, Wp);
    return Tensor({N, C, Ho, Wo}, yptr, nout, x.device);
  }
#endif
  st.argmax_cuda.reset();
  st.argmax_mlx_u32.reset();
  st.argmax_len = 0;
  std::vector<double> xh = x.get_data();
  std::vector<double> yh;
  maxpool2d_forward(xh, N, C, H, W, kH, kW, sh, sw, ph, pw, yh, st.argmax_cpu, Ho, Wo);
  return Tensor::from_data({N, C, Ho, Wo}, yh, x.get_device_type(), x.get_device_index());
}

Tensor maxpool2d_backward_tensor(const Tensor& dy, const MaxPool2dFwdState& st, size_t N, size_t C, size_t H,
                                 size_t W, size_t kH, size_t kW, size_t sh, size_t sw, size_t ph, size_t pw,
                                 size_t Ho, size_t Wo) {
  (void)kH;
  (void)kW;
  (void)sh;
  (void)sw;
#if defined(WITH_CUDA)
  if (dy.device.type == DeviceType::CUDA && st.argmax_cuda) {
    const size_t Hp = H + 2 * ph;
    const size_t Wp = W + 2 * pw;
    double* dxp = backend_alloc(dy.device, N * C * H * W);
    gpu_maxpool2d_backward_nchw(dy.data, static_cast<const unsigned long long*>(st.argmax_cuda.get()), dxp, N, C, H, W,
                                ph, pw, Ho, Wo, Hp, Wp);
    return Tensor({N, C, H, W}, dxp, N * C * H * W, dy.device);
  }
#endif
#if defined(WITH_MLX)
  if (dy.device.type == DeviceType::MLX && st.argmax_mlx_u32) {
    const size_t Hp = H + 2 * ph;
    const size_t Wp = W + 2 * pw;
    double* dxp = backend_alloc(dy.device, N * C * H * W);
    mlx_maxpool2d_backward_nchw(dy.data, static_cast<const uint32_t*>(st.argmax_mlx_u32.get()), dxp, N, C, H, W, ph, pw,
                                Ho, Wo, Hp, Wp);
    return Tensor({N, C, H, W}, dxp, N * C * H * W, dy.device);
  }
#endif
  std::vector<double> dyh = dy.get_data();
  std::vector<double> dx;
  maxpool2d_backward(dyh, st.argmax_cpu, N, C, H, W, ph, pw, Ho, Wo, dx);
  return Tensor::from_data({N, C, H, W}, dx, dy.get_device_type(), dy.get_device_index());
}

Tensor avgpool2d_forward_tensor(const Tensor& x, size_t kH, size_t kW, size_t sh, size_t sw, size_t ph, size_t pw,
                                size_t& Ho, size_t& Wo) {
  const auto xd = x.get_dims();
  if (xd.size() != 4) {
    throw std::runtime_error("avgpool2d_forward_tensor: NCHW required");
  }
  const size_t N = xd[0], C = xd[1], H = xd[2], W = xd[3];
  Ho = conv_out_dim_floor(H, kH, sh, ph);
  Wo = conv_out_dim_floor(W, kW, sw, pw);
#if defined(WITH_CUDA)
  if (x.device.type == DeviceType::CUDA) {
    const size_t Hp = H + 2 * ph;
    const size_t Wp = W + 2 * pw;
    const size_t nout = N * C * Ho * Wo;
    double* yptr = backend_alloc(x.device, nout);
    gpu_avgpool2d_forward_nchw(x.data, yptr, N, C, H, W, kH, kW, sh, sw, ph, pw, Ho, Wo, Hp, Wp);
    return Tensor({N, C, Ho, Wo}, yptr, nout, x.device);
  }
#endif
#if defined(WITH_MLX)
  if (x.device.type == DeviceType::MLX) {
    const size_t Hp = H + 2 * ph;
    const size_t Wp = W + 2 * pw;
    const size_t nout = N * C * Ho * Wo;
    double* yptr = backend_alloc(x.device, nout);
    mlx_avgpool2d_forward_nchw(x.data, yptr, N, C, H, W, kH, kW, sh, sw, ph, pw, Ho, Wo, Hp, Wp);
    return Tensor({N, C, Ho, Wo}, yptr, nout, x.device);
  }
#endif
  std::vector<double> xh = x.get_data();
  std::vector<double> yh;
  avgpool2d_forward(xh, N, C, H, W, kH, kW, sh, sw, ph, pw, yh, Ho, Wo);
  return Tensor::from_data({N, C, Ho, Wo}, yh, x.get_device_type(), x.get_device_index());
}

Tensor avgpool2d_backward_tensor(const Tensor& dy, size_t N, size_t C, size_t H, size_t W, size_t kH, size_t kW,
                                 size_t sh, size_t sw, size_t ph, size_t pw, size_t Ho, size_t Wo) {
#if defined(WITH_CUDA)
  if (dy.device.type == DeviceType::CUDA) {
    const size_t Hp = H + 2 * ph;
    const size_t Wp = W + 2 * pw;
    double* dxp = backend_alloc(dy.device, N * C * H * W);
    gpu_avgpool2d_backward_nchw(dy.data, dxp, N, C, H, W, kH, kW, sh, sw, ph, pw, Ho, Wo, Hp, Wp);
    return Tensor({N, C, H, W}, dxp, N * C * H * W, dy.device);
  }
#endif
#if defined(WITH_MLX)
  if (dy.device.type == DeviceType::MLX) {
    const size_t Hp = H + 2 * ph;
    const size_t Wp = W + 2 * pw;
    double* dxp = backend_alloc(dy.device, N * C * H * W);
    mlx_avgpool2d_backward_nchw(dy.data, dxp, N, C, H, W, kH, kW, sh, sw, ph, pw, Ho, Wo, Hp, Wp);
    return Tensor({N, C, H, W}, dxp, N * C * H * W, dy.device);
  }
#endif
  std::vector<double> dyh = dy.get_data();
  std::vector<double> dx;
  avgpool2d_backward(dyh, N, C, H, W, kH, kW, sh, sw, ph, pw, Ho, Wo, dx);
  return Tensor::from_data({N, C, H, W}, dx, dy.get_device_type(), dy.get_device_index());
}
