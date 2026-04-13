#include "tensor_kernels.h"
#include "apple_backend_config.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

#ifdef WITH_OPENMP
#include <omp.h>
#endif

namespace {
[[noreturn]] void throw_cuda_unavailable() {
  throw std::runtime_error("CUDA backend requested but this binary was built without CUDA support");
}
#ifndef WITH_MLX
[[noreturn]] void throw_mlx_unavailable() {
  throw std::runtime_error("MLX backend requested but this binary was built without WITH_MLX support");
}
#endif
[[noreturn]] void throw_mps_unavailable() {
  throw std::runtime_error("MPS backend is not implemented; use device=\"mlx\" for Apple GPU path");
}

void ensure_cpu_ptr(double* ptr) {
  if (!ptr) {
    throw std::runtime_error("CPU allocation failed");
  }
}
}  // namespace

Device parse_device(const std::string& device_name, int index) {
  if (device_name == "cpu" || device_name == "CPU") {
    return Device{DeviceType::CPU, index};
  }
  if (device_name == "mlx" || device_name == "MLX") {
    return Device{DeviceType::MLX, index};
  }
  if (device_name == "mps" || device_name == "MPS" || device_name == "metal" || device_name == "METAL") {
    return Device{DeviceType::MPS, index};
  }
  if (device_name == "cuda" || device_name == "CUDA" || device_name == "gpu" || device_name == "GPU") {
    return Device{DeviceType::CUDA, index};
  }
  throw std::runtime_error("Unknown device type: " + device_name);
}

const char* device_type_name(DeviceType type) {
  switch (type) {
    case DeviceType::CPU:
      return "cpu";
    case DeviceType::CUDA:
      return "cuda";
    case DeviceType::MLX:
      return "mlx";
    case DeviceType::MPS:
      return "mps";
    default:
      return "unknown";
  }
}

bool device_equal(const Device& a, const Device& b) {
  return a.type == b.type && a.index == b.index;
}

bool backend_mlx_native_available() {
#ifdef WITH_MLX
  return mlx_native_available();
#else
  return false;
#endif
}

size_t backend_mlx_dispatch_count() {
#ifdef WITH_MLX
  return mlx_dispatch_count();
#else
  return 0;
#endif
}

void backend_mlx_reset_dispatch_count() {
#ifdef WITH_MLX
  mlx_reset_dispatch_count();
#endif
}

double* backend_alloc(const Device& device, size_t n) {
  if (device.type == DeviceType::CPU) {
    double* ptr = new double[n];
    ensure_cpu_ptr(ptr);
    return ptr;
  }
  if (device.type == DeviceType::MLX) {
#ifdef WITH_MLX
    return mlx_alloc(n);
#else
    throw_mlx_unavailable();
#endif
  }
  if (device.type == DeviceType::MPS) {
    throw_mps_unavailable();
  }
#ifdef WITH_CUDA
  return gpu_alloc(n);
#else
  (void)n;
  throw_cuda_unavailable();
  return nullptr;
#endif
}

void backend_free(const Device& device, double* ptr) {
  if (!ptr) return;
  if (device.type == DeviceType::CPU) {
    delete[] ptr;
    return;
  }
  if (device.type == DeviceType::MLX) {
#ifdef WITH_MLX
    mlx_free(ptr);
    return;
#else
    throw_mlx_unavailable();
#endif
  }
  if (device.type == DeviceType::MPS) {
    throw_mps_unavailable();
  }
#ifdef WITH_CUDA
  gpu_free(ptr);
#else
  throw_cuda_unavailable();
#endif
}

void backend_upload(const Device& device, double* dst, const double* src, size_t n) {
  if (device.type == DeviceType::CPU) {
    std::memcpy(dst, src, n * sizeof(double));
    return;
  }
  if (device.type == DeviceType::MLX) {
#ifdef WITH_MLX
    mlx_upload(dst, src, n);
    return;
#else
    throw_mlx_unavailable();
#endif
  }
  if (device.type == DeviceType::MPS) {
    throw_mps_unavailable();
  }
#ifdef WITH_CUDA
  gpu_upload(dst, src, n);
#else
  throw_cuda_unavailable();
#endif
}

void backend_download(const Device& device, double* dst, const double* src, size_t n) {
  if (device.type == DeviceType::CPU) {
    std::memcpy(dst, src, n * sizeof(double));
    return;
  }
  if (device.type == DeviceType::MLX) {
#ifdef WITH_MLX
    mlx_download(dst, src, n);
    return;
#else
    throw_mlx_unavailable();
#endif
  }
  if (device.type == DeviceType::MPS) {
    throw_mps_unavailable();
  }
#ifdef WITH_CUDA
  gpu_download(dst, src, n);
#else
  throw_cuda_unavailable();
#endif
}

void backend_copy_device(const Device& device, double* dst, const double* src, size_t n) {
  if (device.type == DeviceType::CPU) {
    std::memcpy(dst, src, n * sizeof(double));
    return;
  }
  if (device.type == DeviceType::MLX) {
#ifdef WITH_MLX
    mlx_copy_device(dst, src, n);
    return;
#else
    throw_mlx_unavailable();
#endif
  }
  if (device.type == DeviceType::MPS) {
    throw_mps_unavailable();
  }
#ifdef WITH_CUDA
  gpu_copy_device(dst, src, n);
#else
  throw_cuda_unavailable();
#endif
}

void backend_zero(const Device& device, double* ptr, size_t n) {
  if (device.type == DeviceType::CPU) {
    std::fill(ptr, ptr + n, 0.0);
    return;
  }
  if (device.type == DeviceType::MLX) {
#ifdef WITH_MLX
    mlx_zero(ptr, n);
    return;
#else
    throw_mlx_unavailable();
#endif
  }
  if (device.type == DeviceType::MPS) {
    throw_mps_unavailable();
  }
#ifdef WITH_CUDA
  gpu_zero(ptr, n);
#else
  throw_cuda_unavailable();
#endif
}

void backend_sync(const Device& device) {
  if (device.type == DeviceType::CPU) return;
  if (device.type == DeviceType::MLX) {
#ifdef WITH_MLX
    mlx_sync();
    return;
#else
    throw_mlx_unavailable();
#endif
  }
  if (device.type == DeviceType::MPS) {
    throw_mps_unavailable();
  }
#ifdef WITH_CUDA
  gpu_sync();
#else
  throw_cuda_unavailable();
#endif
}

void backend_neg(const Device& device, const double* a, double* out, size_t n) {
  if (device.type == DeviceType::CPU) {
#ifdef WITH_OPENMP
#pragma omp parallel for if (n > 8192)
#endif
    for (size_t i = 0; i < n; ++i) out[i] = -a[i];
    return;
  }
  if (device.type == DeviceType::MLX) {
#ifdef WITH_MLX
    mlx_neg(a, out, n);
    return;
#else
    throw_mlx_unavailable();
#endif
  }
  if (device.type == DeviceType::MPS) throw_mps_unavailable();
#ifdef WITH_CUDA
  gpu_neg(a, out, n);
#else
  throw_cuda_unavailable();
#endif
}

void backend_reciprocal(const Device& device, const double* a, double* out, size_t n) {
  if (device.type == DeviceType::CPU) {
#ifdef WITH_OPENMP
#pragma omp parallel for if (n > 8192)
#endif
    for (size_t i = 0; i < n; ++i) out[i] = 1.0 / a[i];
    return;
  }
  if (device.type == DeviceType::MLX) {
#ifdef WITH_MLX
    mlx_reciprocal(a, out, n);
    return;
#else
    throw_mlx_unavailable();
#endif
  }
  if (device.type == DeviceType::MPS) throw_mps_unavailable();
#ifdef WITH_CUDA
  gpu_reciprocal(a, out, n);
#else
  throw_cuda_unavailable();
#endif
}

void backend_add(const Device& device, const double* a, const double* b, double* out, size_t n) {
  if (device.type == DeviceType::CPU) {
#ifdef WITH_OPENMP
#pragma omp parallel for if (n > 8192)
#endif
    for (size_t i = 0; i < n; ++i) out[i] = a[i] + b[i];
    return;
  }
  if (device.type == DeviceType::MLX) {
#ifdef WITH_MLX
    mlx_add(a, b, out, n);
    return;
#else
    throw_mlx_unavailable();
#endif
  }
  if (device.type == DeviceType::MPS) throw_mps_unavailable();
#ifdef WITH_CUDA
  gpu_add_impl(a, b, out, n);
#else
  throw_cuda_unavailable();
#endif
}

void backend_subtract(const Device& device, const double* a, const double* b, double* out, size_t n) {
  if (device.type == DeviceType::CPU) {
#ifdef WITH_OPENMP
#pragma omp parallel for if (n > 8192)
#endif
    for (size_t i = 0; i < n; ++i) out[i] = a[i] - b[i];
    return;
  }
  if (device.type == DeviceType::MLX) {
#ifdef WITH_MLX
    mlx_subtract(a, b, out, n);
    return;
#else
    throw_mlx_unavailable();
#endif
  }
  if (device.type == DeviceType::MPS) throw_mps_unavailable();
#ifdef WITH_CUDA
  gpu_subtract(a, b, out, n);
#else
  throw_cuda_unavailable();
#endif
}

void backend_mult_scalar(const Device& device, const double* a, double s, double* out, size_t n) {
  if (device.type == DeviceType::CPU) {
#ifdef WITH_OPENMP
#pragma omp parallel for if (n > 8192)
#endif
    for (size_t i = 0; i < n; ++i) out[i] = a[i] * s;
    return;
  }
  if (device.type == DeviceType::MLX) {
#ifdef WITH_MLX
    mlx_mult_scalar(a, s, out, n);
    return;
#else
    throw_mlx_unavailable();
#endif
  }
  if (device.type == DeviceType::MPS) throw_mps_unavailable();
#ifdef WITH_CUDA
  gpu_mult_scalar(a, s, out, n);
#else
  throw_cuda_unavailable();
#endif
}

void backend_elementwise_mult(const Device& device, const double* a, const double* b, double* out, size_t n) {
  if (device.type == DeviceType::CPU) {
#ifdef WITH_OPENMP
#pragma omp parallel for if (n > 8192)
#endif
    for (size_t i = 0; i < n; ++i) out[i] = a[i] * b[i];
    return;
  }
  if (device.type == DeviceType::MLX) {
#ifdef WITH_MLX
    mlx_elementwise_mult(a, b, out, n);
    return;
#else
    throw_mlx_unavailable();
#endif
  }
  if (device.type == DeviceType::MPS) throw_mps_unavailable();
#ifdef WITH_CUDA
  gpu_elementwise_mult(a, b, out, n);
#else
  throw_cuda_unavailable();
#endif
}

void backend_pow(const Device& device, const double* a, double p, double* out, size_t n) {
  if (device.type == DeviceType::CPU) {
#ifdef WITH_OPENMP
#pragma omp parallel for if (n > 8192)
#endif
    for (size_t i = 0; i < n; ++i) out[i] = std::pow(a[i], p);
    return;
  }
  if (device.type == DeviceType::MLX) {
#ifdef WITH_MLX
    mlx_pow(a, p, out, n);
    return;
#else
    throw_mlx_unavailable();
#endif
  }
  if (device.type == DeviceType::MPS) throw_mps_unavailable();
#ifdef WITH_CUDA
  gpu_pow(a, p, out, n);
#else
  throw_cuda_unavailable();
#endif
}

void backend_relu(const Device& device, const double* a, double* out, size_t n) {
  if (device.type == DeviceType::CPU) {
#ifdef WITH_OPENMP
#pragma omp parallel for if (n > 8192)
#endif
    for (size_t i = 0; i < n; ++i) out[i] = a[i] > 0.0 ? a[i] : 0.0;
    return;
  }
  if (device.type == DeviceType::MLX) {
#ifdef WITH_MLX
    mlx_relu(a, out, n);
    return;
#else
    throw_mlx_unavailable();
#endif
  }
  if (device.type == DeviceType::MPS) throw_mps_unavailable();
#ifdef WITH_CUDA
  gpu_relu(a, out, n);
#else
  throw_cuda_unavailable();
#endif
}

void backend_binarilize(const Device& device, const double* a, double* out, size_t n) {
  if (device.type == DeviceType::CPU) {
#ifdef WITH_OPENMP
#pragma omp parallel for if (n > 8192)
#endif
    for (size_t i = 0; i < n; ++i) out[i] = a[i] > 0.0 ? 1.0 : 0.0;
    return;
  }
  if (device.type == DeviceType::MLX) {
#ifdef WITH_MLX
    mlx_binarilize(a, out, n);
    return;
#else
    throw_mlx_unavailable();
#endif
  }
  if (device.type == DeviceType::MPS) throw_mps_unavailable();
#ifdef WITH_CUDA
  gpu_binarilize(a, out, n);
#else
  throw_cuda_unavailable();
#endif
}

void backend_exp(const Device& device, const double* a, double* out, size_t n) {
  if (device.type == DeviceType::CPU) {
#ifdef WITH_OPENMP
#pragma omp parallel for if (n > 8192)
#endif
    for (size_t i = 0; i < n; ++i) out[i] = std::exp(a[i]);
    return;
  }
  if (device.type == DeviceType::MLX) {
#ifdef WITH_MLX
    mlx_exp(a, out, n);
    return;
#else
    throw_mlx_unavailable();
#endif
  }
  if (device.type == DeviceType::MPS) throw_mps_unavailable();
#ifdef WITH_CUDA
  gpu_exp(a, out, n);
#else
  throw_cuda_unavailable();
#endif
}

void backend_sigmoid(const Device& device, const double* a, double* out, size_t n) {
  if (device.type == DeviceType::CPU) {
#ifdef WITH_OPENMP
#pragma omp parallel for if (n > 8192)
#endif
    for (size_t i = 0; i < n; ++i) out[i] = 1.0 / (1.0 + std::exp(-a[i]));
    return;
  }
  if (device.type == DeviceType::MLX) {
#ifdef WITH_MLX
    mlx_sigmoid(a, out, n);
    return;
#else
    throw_mlx_unavailable();
#endif
  }
  if (device.type == DeviceType::MPS) throw_mps_unavailable();
#ifdef WITH_CUDA
  gpu_sigmoid(a, out, n);
#else
  throw_cuda_unavailable();
#endif
}

void backend_tanh(const Device& device, const double* a, double* out, size_t n) {
  if (device.type == DeviceType::CPU) {
#ifdef WITH_OPENMP
#pragma omp parallel for if (n > 8192)
#endif
    for (size_t i = 0; i < n; ++i) out[i] = std::tanh(a[i]);
    return;
  }
  if (device.type == DeviceType::MLX) {
#ifdef WITH_MLX
    mlx_tanh(a, out, n);
    return;
#else
    throw_mlx_unavailable();
#endif
  }
  if (device.type == DeviceType::MPS) throw_mps_unavailable();
#ifdef WITH_CUDA
  gpu_tanh(a, out, n);
#else
  throw_cuda_unavailable();
#endif
}

void backend_softmax_last_dim(const Device& device, const double* a, double* out, size_t rows, size_t cols) {
  if (device.type == DeviceType::CPU) {
#ifdef WITH_OPENMP
#pragma omp parallel for if (rows > 64)
#endif
    for (size_t r = 0; r < rows; ++r) {
      size_t base = r * cols;
      double mx = a[base];
      for (size_t j = 1; j < cols; ++j) mx = std::max(mx, a[base + j]);
      double sum = 0.0;
      for (size_t j = 0; j < cols; ++j) {
        out[base + j] = std::exp(a[base + j] - mx);
        sum += out[base + j];
      }
      double inv = 1.0 / sum;
      for (size_t j = 0; j < cols; ++j) out[base + j] *= inv;
    }
    return;
  }
  if (device.type == DeviceType::MLX) {
#ifdef WITH_MLX
    mlx_softmax_last_dim(a, out, rows, cols);
    return;
#else
    throw_mlx_unavailable();
#endif
  }
  if (device.type == DeviceType::MPS) throw_mps_unavailable();
#ifdef WITH_CUDA
  gpu_softmax_last_dim(a, out, rows, cols);
#else
  throw_cuda_unavailable();
#endif
}

void backend_sum_all(const Device& device, const double* a, double* out, size_t n) {
  if (device.type == DeviceType::CPU) {
    double s = 0.0;
#ifdef WITH_OPENMP
#pragma omp parallel for reduction(+ : s) if (n > 8192)
#endif
    for (size_t i = 0; i < n; ++i) s += a[i];
    out[0] = s;
    return;
  }
  if (device.type == DeviceType::MLX) {
#ifdef WITH_MLX
    mlx_sum_all(a, out, n);
    return;
#else
    throw_mlx_unavailable();
#endif
  }
  if (device.type == DeviceType::MPS) throw_mps_unavailable();
#ifdef WITH_CUDA
  gpu_sum_all(a, out, n);
#else
  throw_cuda_unavailable();
#endif
}

void backend_add_rowwise(const Device& device, const double* a, const double* row, double* out,
                         size_t batch, size_t n) {
  if (device.type == DeviceType::CPU) {
#ifdef WITH_OPENMP
#pragma omp parallel for if (batch * n > 8192)
#endif
    for (size_t i = 0; i < batch; ++i) {
      for (size_t j = 0; j < n; ++j) {
        out[i * n + j] = a[i * n + j] + row[j];
      }
    }
    return;
  }
  if (device.type == DeviceType::MLX) {
#ifdef WITH_MLX
    mlx_add_rowwise(a, row, out, batch, n);
    return;
#else
    throw_mlx_unavailable();
#endif
  }
  if (device.type == DeviceType::MPS) throw_mps_unavailable();
#ifdef WITH_CUDA
  gpu_add_rowwise(a, row, out, batch, n);
#else
  throw_cuda_unavailable();
#endif
}

void backend_transpose_2d(const Device& device, const double* a, double* out, size_t rows, size_t cols) {
  if (device.type == DeviceType::CPU) {
#ifdef WITH_OPENMP
#pragma omp parallel for if (rows * cols > 8192)
#endif
    for (size_t i = 0; i < rows; ++i) {
      for (size_t j = 0; j < cols; ++j) {
        out[j * rows + i] = a[i * cols + j];
      }
    }
    return;
  }
  if (device.type == DeviceType::MLX) {
#ifdef WITH_MLX
    mlx_transpose_2d(a, out, rows, cols);
    return;
#else
    throw_mlx_unavailable();
#endif
  }
  if (device.type == DeviceType::MPS) throw_mps_unavailable();
#ifdef WITH_CUDA
  gpu_transpose_2d(a, out, rows, cols);
#else
  throw_cuda_unavailable();
#endif
}

void backend_transpose_3d(const Device& device, const double* a, double* out, size_t B, size_t R, size_t C) {
  if (device.type == DeviceType::CPU) {
    for (size_t b = 0; b < B; ++b) {
      for (size_t i = 0; i < R; ++i) {
        for (size_t j = 0; j < C; ++j) {
          out[b * C * R + j * R + i] = a[b * R * C + i * C + j];
        }
      }
    }
    return;
  }
  if (device.type == DeviceType::MLX) {
#ifdef WITH_MLX
    mlx_transpose_3d(a, out, B, R, C);
    return;
#else
    throw_mlx_unavailable();
#endif
  }
  if (device.type == DeviceType::MPS) throw_mps_unavailable();
#ifdef WITH_CUDA
  gpu_transpose_3d(a, out, B, R, C);
#else
  throw_cuda_unavailable();
#endif
}

void backend_matmul_2d(const Device& device, const double* A, const double* B, double* C,
                       size_t M, size_t K, size_t N) {
  if (device.type == DeviceType::CPU) {
#ifdef WITH_OPENMP
#pragma omp parallel for if (M * N * K > 500000)
#endif
    for (size_t i = 0; i < M; ++i) {
      for (size_t k = 0; k < K; ++k) {
        const double a = A[i * K + k];
        for (size_t j = 0; j < N; ++j) {
          C[i * N + j] += a * B[k * N + j];
        }
      }
    }
    return;
  }
  if (device.type == DeviceType::MLX) {
#ifdef WITH_MLX
    mlx_matmul_2d(A, B, C, M, K, N);
    return;
#else
    throw_mlx_unavailable();
#endif
  }
  if (device.type == DeviceType::MPS) throw_mps_unavailable();
#ifdef WITH_CUDA
  gpu_matmul_2d(A, B, C, M, K, N);
#else
  throw_cuda_unavailable();
#endif
}

void backend_matmul_batched(const Device& device, const double* A, const double* B, double* C,
                            size_t batch, size_t M, size_t K, size_t N) {
  if (device.type == DeviceType::CPU) {
#ifdef WITH_OPENMP
#pragma omp parallel for if (batch > 1 && batch * M * N * K > 500000)
#endif
    for (size_t b = 0; b < batch; ++b) {
      const double* A_b = A + b * M * K;
      double* C_b = C + b * M * N;
      for (size_t i = 0; i < M; ++i) {
        for (size_t k = 0; k < K; ++k) {
          const double a = A_b[i * K + k];
          for (size_t j = 0; j < N; ++j) {
            C_b[i * N + j] += a * B[k * N + j];
          }
        }
      }
    }
    return;
  }
  if (device.type == DeviceType::MLX) {
#ifdef WITH_MLX
    mlx_matmul_batched(A, B, C, batch, M, K, N);
    return;
#else
    throw_mlx_unavailable();
#endif
  }
  if (device.type == DeviceType::MPS) throw_mps_unavailable();
#ifdef WITH_CUDA
  gpu_matmul_batched(A, B, C, batch, M, K, N);
#else
  throw_cuda_unavailable();
#endif
}
