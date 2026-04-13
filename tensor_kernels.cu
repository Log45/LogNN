#include "tensor_kernels.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <stdexcept>

// ============================================================
// Error checking
// ============================================================
#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    cudaError_t err = (call);                                                \
    if (err != cudaSuccess) {                                                \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,      \
              cudaGetErrorString(err));                                       \
      throw std::runtime_error(cudaGetErrorString(err));                     \
    }                                                                        \
  } while (0)

// Check for kernel launch errors (async - checks cudaGetLastError)
#define KERNEL_CHECK()                                                       \
  do {                                                                       \
    cudaError_t err = cudaGetLastError();                                    \
    if (err != cudaSuccess) {                                                \
      fprintf(stderr, "Kernel launch error at %s:%d: %s\n",                 \
              __FILE__, __LINE__, cudaGetErrorString(err));                   \
      throw std::runtime_error(cudaGetErrorString(err));                     \
    }                                                                        \
  } while (0)

// ============================================================
// GPU memory helpers
// ============================================================
double* gpu_alloc(size_t n) {
  double* ptr = nullptr;
  CUDA_CHECK(cudaMalloc(&ptr, n * sizeof(double)));
  return ptr;
}

void gpu_free(double* ptr) {
  if (ptr) cudaFree(ptr);
}

void gpu_upload(double* dst, const double* src, size_t n) {
  CUDA_CHECK(cudaMemcpy(dst, src, n * sizeof(double), cudaMemcpyHostToDevice));
}

void gpu_download(double* dst, const double* src, size_t n) {
  CUDA_CHECK(cudaMemcpy(dst, src, n * sizeof(double), cudaMemcpyDeviceToHost));
}

void gpu_copy_device(double* dst, const double* src, size_t n) {
  CUDA_CHECK(cudaMemcpy(dst, src, n * sizeof(double), cudaMemcpyDeviceToDevice));
}

void gpu_zero(double* ptr, size_t n) {
  CUDA_CHECK(cudaMemset(ptr, 0, n * sizeof(double)));
}

void gpu_sync() {
  CUDA_CHECK(cudaDeviceSynchronize());
}

// ============================================================
// Element-wise kernels
// ============================================================
static const int BLOCK_SIZE = 256;

static inline int grid1d(size_t n) {
  return (int)((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
}

__global__ void kernel_neg(const double* a, double* out, size_t n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = -a[i];
}

__global__ void kernel_reciprocal(const double* a, double* out, size_t n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = 1.0 / a[i];
}

__global__ void kernel_add(const double* a, const double* b, double* out, size_t n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = a[i] + b[i];
}

__global__ void kernel_subtract(const double* a, const double* b, double* out, size_t n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = a[i] - b[i];
}

__global__ void kernel_mult_scalar(const double* a, double s, double* out, size_t n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = a[i] * s;
}

__global__ void kernel_elementwise_mult(const double* a, const double* b, double* out, size_t n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = a[i] * b[i];
}

__global__ void kernel_pow(const double* a, double p, double* out, size_t n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = pow(a[i], p);
}

__global__ void kernel_relu(const double* a, double* out, size_t n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = a[i] > 0.0 ? a[i] : 0.0;
}

__global__ void kernel_binarilize(const double* a, double* out, size_t n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = a[i] > 0.0 ? 1.0 : 0.0;
}

__global__ void kernel_exp(const double* a, double* out, size_t n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = exp(a[i]);
}

__global__ void kernel_sigmoid(const double* a, double* out, size_t n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    double x = a[i];
    out[i] = 1.0 / (1.0 + exp(-x));
  }
}

__global__ void kernel_tanh(const double* a, double* out, size_t n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = tanh(a[i]);
}

// One thread per row: stable softmax along last dimension (cols).
__global__ void kernel_softmax_last_dim(const double* a, double* out, size_t rows, size_t cols) {
  size_t r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= rows) return;
  size_t base = r * cols;
  double mx = a[base];
  for (size_t j = 1; j < cols; ++j) {
    double v = a[base + j];
    if (v > mx) mx = v;
  }
  double sum = 0.0;
  for (size_t j = 0; j < cols; ++j) {
    double e = exp(a[base + j] - mx);
    out[base + j] = e;
    sum += e;
  }
  double inv = 1.0 / sum;
  for (size_t j = 0; j < cols; ++j) out[base + j] *= inv;
}

__global__ void kernel_sum_atomic(const double* a, double* out, size_t n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    atomicAdd(out, a[i]);
  }
}

// out[b*n + j] = a[b*n + j] + row[j], row length n (typically batch 1 x n).
__global__ void kernel_add_rowwise(const double* a, const double* row, double* out, size_t batch, size_t n) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t total = batch * n;
  if (idx < total) {
    size_t j = idx % n;
    out[idx] = a[idx] + row[j];
  }
}

__global__ void kernel_transpose_2d(const double* a, double* out,
                                     size_t rows, size_t cols) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < rows * cols) {
    size_t i = idx / cols;
    size_t j = idx % cols;
    out[j * rows + i] = a[i * cols + j];
  }
}

__global__ void kernel_transpose_3d(const double* a, double* out,
                                     size_t B, size_t R, size_t C) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < B * R * C) {
    size_t b = idx / (R * C);
    size_t rem = idx % (R * C);
    size_t i = rem / C;
    size_t j = rem % C;
    out[b * C * R + j * R + i] = a[idx];
  }
}

// ============================================================
// Tiled matrix multiplication kernel
// C[M,N] = A[M,K] * B[K,N]
// ============================================================
#define TILE 16

__global__ void kernel_matmul(const double* A, const double* B, double* C,
                              size_t M, size_t K, size_t N) {
  __shared__ double As[TILE][TILE];
  __shared__ double Bs[TILE][TILE];

  size_t row = blockIdx.y * TILE + threadIdx.y;
  size_t col = blockIdx.x * TILE + threadIdx.x;

  double sum = 0.0;

  for (size_t t = 0; t < (K + TILE - 1) / TILE; ++t) {
    size_t aCol = t * TILE + threadIdx.x;
    size_t bRow = t * TILE + threadIdx.y;

    As[threadIdx.y][threadIdx.x] = (row < M && aCol < K) ? A[row * K + aCol] : 0.0;
    Bs[threadIdx.y][threadIdx.x] = (bRow < K && col < N) ? B[bRow * N + col] : 0.0;

    __syncthreads();

    #pragma unroll
    for (int k = 0; k < TILE; ++k) {
      sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
    }

    __syncthreads();
  }

  if (row < M && col < N) {
    C[row * N + col] = sum;
  }
}

__global__ void kernel_matmul_batched(const double* A, const double* B, double* C,
                                       size_t M, size_t K, size_t N) {
  __shared__ double As[TILE][TILE];
  __shared__ double Bs[TILE][TILE];

  size_t b = blockIdx.z;
  size_t row = blockIdx.y * TILE + threadIdx.y;
  size_t col = blockIdx.x * TILE + threadIdx.x;

  const double* A_b = A + b * M * K;
  double* C_b = C + b * M * N;

  double sum = 0.0;

  for (size_t t = 0; t < (K + TILE - 1) / TILE; ++t) {
    size_t aCol = t * TILE + threadIdx.x;
    size_t bRow = t * TILE + threadIdx.y;

    As[threadIdx.y][threadIdx.x] = (row < M && aCol < K) ? A_b[row * K + aCol] : 0.0;
    Bs[threadIdx.y][threadIdx.x] = (bRow < K && col < N) ? B[bRow * N + col] : 0.0;

    __syncthreads();

    #pragma unroll
    for (int k = 0; k < TILE; ++k) {
      sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
    }

    __syncthreads();
  }

  if (row < M && col < N) {
    C_b[row * N + col] = sum;
  }
}

// ============================================================
// Dispatch functions - all check for kernel launch errors
// ============================================================
void gpu_neg(const double* a, double* out, size_t n) {
  kernel_neg<<<grid1d(n), BLOCK_SIZE>>>(a, out, n);
  KERNEL_CHECK();
}

void gpu_reciprocal(const double* a, double* out, size_t n) {
  kernel_reciprocal<<<grid1d(n), BLOCK_SIZE>>>(a, out, n);
  KERNEL_CHECK();
}

void gpu_add_impl(const double* a, const double* b, double* out, size_t n) {
  kernel_add<<<grid1d(n), BLOCK_SIZE>>>(a, b, out, n);
  KERNEL_CHECK();
}

void gpu_subtract(const double* a, const double* b, double* out, size_t n) {
  kernel_subtract<<<grid1d(n), BLOCK_SIZE>>>(a, b, out, n);
  KERNEL_CHECK();
}

void gpu_mult_scalar(const double* a, double s, double* out, size_t n) {
  kernel_mult_scalar<<<grid1d(n), BLOCK_SIZE>>>(a, s, out, n);
  KERNEL_CHECK();
}

void gpu_elementwise_mult(const double* a, const double* b, double* out, size_t n) {
  kernel_elementwise_mult<<<grid1d(n), BLOCK_SIZE>>>(a, b, out, n);
  KERNEL_CHECK();
}

void gpu_pow(const double* a, double p, double* out, size_t n) {
  kernel_pow<<<grid1d(n), BLOCK_SIZE>>>(a, p, out, n);
  KERNEL_CHECK();
}

void gpu_relu(const double* a, double* out, size_t n) {
  kernel_relu<<<grid1d(n), BLOCK_SIZE>>>(a, out, n);
  KERNEL_CHECK();
}

void gpu_binarilize(const double* a, double* out, size_t n) {
  kernel_binarilize<<<grid1d(n), BLOCK_SIZE>>>(a, out, n);
  KERNEL_CHECK();
}

void gpu_exp(const double* a, double* out, size_t n) {
  kernel_exp<<<grid1d(n), BLOCK_SIZE>>>(a, out, n);
  KERNEL_CHECK();
}

void gpu_sigmoid(const double* a, double* out, size_t n) {
  kernel_sigmoid<<<grid1d(n), BLOCK_SIZE>>>(a, out, n);
  KERNEL_CHECK();
}

void gpu_tanh(const double* a, double* out, size_t n) {
  kernel_tanh<<<grid1d(n), BLOCK_SIZE>>>(a, out, n);
  KERNEL_CHECK();
}

void gpu_softmax_last_dim(const double* a, double* out, size_t rows, size_t cols) {
  int grid = grid1d(rows);
  kernel_softmax_last_dim<<<grid, BLOCK_SIZE>>>(a, out, rows, cols);
  KERNEL_CHECK();
}

void gpu_sum_all(const double* a, double* out, size_t n) {
  CUDA_CHECK(cudaMemset(out, 0, sizeof(double)));
  kernel_sum_atomic<<<grid1d(n), BLOCK_SIZE>>>(a, out, n);
  KERNEL_CHECK();
}

void gpu_add_rowwise(const double* a, const double* row, double* out, size_t batch, size_t n) {
  size_t total = batch * n;
  kernel_add_rowwise<<<grid1d(total), BLOCK_SIZE>>>(a, row, out, batch, n);
  KERNEL_CHECK();
}

void gpu_transpose_2d(const double* a, double* out, size_t rows, size_t cols) {
  size_t n = rows * cols;
  kernel_transpose_2d<<<grid1d(n), BLOCK_SIZE>>>(a, out, rows, cols);
  KERNEL_CHECK();
}

void gpu_transpose_3d(const double* a, double* out, size_t B, size_t R, size_t C) {
  size_t n = B * R * C;
  kernel_transpose_3d<<<grid1d(n), BLOCK_SIZE>>>(a, out, B, R, C);
  KERNEL_CHECK();
}

void gpu_matmul_2d(const double* A, const double* B, double* C,
                   size_t M, size_t K, size_t N) {
  dim3 block(TILE, TILE);
  dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
  kernel_matmul<<<grid, block>>>(A, B, C, M, K, N);
  KERNEL_CHECK();
}

void gpu_matmul_batched(const double* A, const double* B, double* C,
                        size_t batch, size_t M, size_t K, size_t N) {
  dim3 block(TILE, TILE);
  dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE, batch);
  kernel_matmul_batched<<<grid, block>>>(A, B, C, M, K, N);
  KERNEL_CHECK();
}
