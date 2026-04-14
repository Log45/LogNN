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

// ============================================================
// Conv2d / pooling / conv transpose (NCHW, double)
// ============================================================

unsigned long long* gpu_alloc_ull(size_t n) {
  unsigned long long* p = nullptr;
  CUDA_CHECK(cudaMalloc(&p, n * sizeof(unsigned long long)));
  return p;
}

void gpu_free_ull(unsigned long long* p) {
  if (p) cudaFree(p);
}

void gpu_download_ull(const unsigned long long* src, unsigned long long* dst, size_t n) {
  CUDA_CHECK(cudaMemcpy(dst, src, n * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
}

void gpu_upload_ull(unsigned long long* dst, const unsigned long long* src, size_t n) {
  CUDA_CHECK(cudaMemcpy(dst, src, n * sizeof(unsigned long long), cudaMemcpyHostToDevice));
}

__global__ void k_nchw_pad(const double* x, double* xp, size_t N, size_t Ci, size_t H, size_t W, size_t ph,
                           size_t pw, size_t Hp, size_t Wp) {
  size_t id = blockIdx.x * blockDim.x + threadIdx.x;
  size_t tot = N * Ci * H * W;
  if (id >= tot) return;
  size_t w0 = id % W;
  size_t t = id / W;
  size_t h0 = t % H;
  t /= H;
  size_t c = t % Ci;
  size_t n = t / Ci;
  size_t dst = ((n * Ci + c) * Hp + (h0 + ph)) * Wp + (w0 + pw);
  xp[dst] = x[id];
}

__global__ void k_im2col(const double* xp, double* col, size_t N, size_t Ci, size_t Hp, size_t Wp, size_t kH,
                         size_t kW, size_t sh, size_t sw, size_t Ho, size_t Wo, size_t M, size_t K) {
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= M * K) return;
  size_t m = tid / K;
  size_t kk = tid % K;
  size_t rem = m % (Ho * Wo);
  size_t n = m / (Ho * Wo);
  size_t oh = rem / Wo;
  size_t ow = rem % Wo;
  size_t kw_i = kk % kW;
  size_t t2 = kk / kW;
  size_t kh_i = t2 % kH;
  size_t ci = t2 / kH;
  size_t ih = oh * sh + kh_i;
  size_t iw = ow * sw + kw_i;
  col[tid] = xp[((n * Ci + ci) * Hp + ih) * Wp + iw];
}

__global__ void k_scatter_y_col(const double* ycol, double* y, size_t N, size_t Co, size_t Ho, size_t Wo, size_t M) {
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= M * Co) return;
  size_t m = tid / Co;
  size_t co = tid % Co;
  double v = ycol[tid];
  size_t rem = m % (Ho * Wo);
  size_t n = m / (Ho * Wo);
  size_t oh = rem / Wo;
  size_t ow = rem % Wo;
  y[((n * Co + co) * Ho + oh) * Wo + ow] = v;
}

__global__ void k_col2im_atomic(const double* col, double* dxp, size_t N, size_t Ci, size_t Hp, size_t Wp, size_t kH,
                                size_t kW, size_t sh, size_t sw, size_t Ho, size_t Wo, size_t M, size_t K) {
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= M * K) return;
  size_t m = tid / K;
  size_t kk = tid % K;
  size_t rem = m % (Ho * Wo);
  size_t n = m / (Ho * Wo);
  size_t oh = rem / Wo;
  size_t ow = rem % Wo;
  size_t kw_i = kk % kW;
  size_t t2 = kk / kW;
  size_t kh_i = t2 % kH;
  size_t ci = t2 / kH;
  size_t ih = oh * sh + kh_i;
  size_t iw = ow * sw + kw_i;
  size_t idx = ((n * Ci + ci) * Hp + ih) * Wp + iw;
  atomicAdd(dxp + idx, col[tid]);
}

__global__ void k_nchw_unpad(const double* dxp, double* dx, size_t N, size_t Ci, size_t H, size_t W, size_t ph,
                             size_t pw, size_t Hp, size_t Wp) {
  size_t id = blockIdx.x * blockDim.x + threadIdx.x;
  size_t tot = N * Ci * H * W;
  if (id >= tot) return;
  size_t w0 = id % W;
  size_t t = id / W;
  size_t h0 = t % H;
  t /= H;
  size_t c = t % Ci;
  size_t n = t / Ci;
  size_t src = ((n * Ci + c) * Hp + (h0 + ph)) * Wp + (w0 + pw);
  dx[id] = dxp[src];
}

void gpu_conv2d_forward_nchw(const double* x, const double* w, double* y, size_t N, size_t Ci, size_t H, size_t W,
                             size_t Co, size_t kH, size_t kW, size_t sh, size_t sw, size_t ph, size_t pw, size_t Ho,
                             size_t Wo) {
  const size_t Hp = H + 2 * ph;
  const size_t Wp = W + 2 * pw;
  const size_t K = Ci * kH * kW;
  const size_t M = N * Ho * Wo;
  double* xp = gpu_alloc(N * Ci * Hp * Wp);
  gpu_zero(xp, N * Ci * Hp * Wp);
  {
    int grid = grid1d(N * Ci * H * W);
    k_nchw_pad<<<grid, BLOCK_SIZE>>>(x, xp, N, Ci, H, W, ph, pw, Hp, Wp);
    KERNEL_CHECK();
  }
  double* col = gpu_alloc(M * K);
  {
    int grid = grid1d(M * K);
    k_im2col<<<grid, BLOCK_SIZE>>>(xp, col, N, Ci, Hp, Wp, kH, kW, sh, sw, Ho, Wo, M, K);
    KERNEL_CHECK();
  }
  gpu_free(xp);
  double* wt = gpu_alloc(K * Co);
  gpu_transpose_2d(w, wt, Co, K);
  double* ycol = gpu_alloc(M * Co);
  gpu_zero(ycol, M * Co);
  gpu_matmul_2d(col, wt, ycol, M, K, Co);
  gpu_free(col);
  gpu_free(wt);
  gpu_zero(y, N * Co * Ho * Wo);
  {
    int grid = grid1d(M * Co);
    k_scatter_y_col<<<grid, BLOCK_SIZE>>>(ycol, y, N, Co, Ho, Wo, M);
    KERNEL_CHECK();
  }
  gpu_free(ycol);
}

__global__ void k_dcol_from_dy_w(const double* dy_nchw, const double* w, double* dcol, size_t M, size_t Co, size_t K,
                                 size_t N, size_t Ho, size_t Wo) {
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= M * K) return;
  size_t m = tid / K;
  size_t k = tid % K;
  size_t rem = m % (Ho * Wo);
  size_t n = m / (Ho * Wo);
  size_t oh = rem / Wo;
  size_t ow = rem % Wo;
  double s = 0.0;
  for (size_t co = 0; co < Co; ++co) {
    double dyv = dy_nchw[((n * Co + co) * Ho + oh) * Wo + ow];
    s += dyv * w[co * K + k];
  }
  dcol[tid] = s;
}

__global__ void k_dw_from_dy_col(const double* dy_nchw, const double* col, double* dw, size_t M, size_t Co, size_t K,
                                 size_t Ho, size_t Wo) {
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= Co * K) return;
  size_t co = tid / K;
  size_t k = tid % K;
  double s = 0.0;
  for (size_t m = 0; m < M; ++m) {
    size_t rem = m % (Ho * Wo);
    size_t n = m / (Ho * Wo);
    size_t oh = rem / Wo;
    size_t ow = rem % Wo;
    double dyv = dy_nchw[((n * Co + co) * Ho + oh) * Wo + ow];
    s += dyv * col[m * K + k];
  }
  dw[tid] = s;
}

void gpu_conv2d_backward_nchw(const double* dy, const double* x, const double* w, double* dx, double* dw, size_t N,
                              size_t Ci, size_t H, size_t W, size_t Co, size_t kH, size_t kW, size_t sh, size_t sw,
                              size_t ph, size_t pw, size_t Ho, size_t Wo) {
  const size_t Hp = H + 2 * ph;
  const size_t Wp = W + 2 * pw;
  const size_t K = Ci * kH * kW;
  const size_t M = N * Ho * Wo;
  double* xp = gpu_alloc(N * Ci * Hp * Wp);
  gpu_zero(xp, N * Ci * Hp * Wp);
  k_nchw_pad<<<grid1d(N * Ci * H * W), BLOCK_SIZE>>>(x, xp, N, Ci, H, W, ph, pw, Hp, Wp);
  KERNEL_CHECK();
  double* col = gpu_alloc(M * K);
  k_im2col<<<grid1d(M * K), BLOCK_SIZE>>>(xp, col, N, Ci, Hp, Wp, kH, kW, sh, sw, Ho, Wo, M, K);
  KERNEL_CHECK();
  gpu_free(xp);
  double* dcol = gpu_alloc(M * K);
  k_dcol_from_dy_w<<<grid1d(M * K), BLOCK_SIZE>>>(dy, w, dcol, M, Co, K, N, Ho, Wo);
  KERNEL_CHECK();
  k_dw_from_dy_col<<<grid1d(Co * K), BLOCK_SIZE>>>(dy, col, dw, M, Co, K, Ho, Wo);
  KERNEL_CHECK();
  double* dxp = gpu_alloc(N * Ci * Hp * Wp);
  gpu_zero(dxp, N * Ci * Hp * Wp);
  k_col2im_atomic<<<grid1d(M * K), BLOCK_SIZE>>>(dcol, dxp, N, Ci, Hp, Wp, kH, kW, sh, sw, Ho, Wo, M, K);
  KERNEL_CHECK();
  gpu_free(dcol);
  gpu_free(col);
  gpu_zero(dx, N * Ci * H * W);
  k_nchw_unpad<<<grid1d(N * Ci * H * W), BLOCK_SIZE>>>(dxp, dx, N, Ci, H, W, ph, pw, Hp, Wp);
  KERNEL_CHECK();
  gpu_free(dxp);
}

__global__ void k_maxpool_fwd(const double* xp, double* y, unsigned long long* argmax, size_t N, size_t C, size_t Hp,
                              size_t Wp, size_t kH, size_t kW, size_t sh, size_t sw, size_t Ho, size_t Wo) {
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  size_t nout = N * C * Ho * Wo;
  if (tid >= nout) return;
  size_t ow = tid % Wo;
  size_t t = tid / Wo;
  size_t oh = t % Ho;
  t /= Ho;
  size_t c = t % C;
  size_t n = t / C;
  double best = -1e300;
  unsigned long long best_i = 0;
  bool first = true;
  for (size_t kh = 0; kh < kH; ++kh) {
    for (size_t kw = 0; kw < kW; ++kw) {
      size_t ih = oh * sh + kh;
      size_t iw = ow * sw + kw;
      size_t li = ((n * C + c) * Hp + ih) * Wp + iw;
      double v = xp[li];
      if (first || v > best) {
        best = v;
        best_i = li;
        first = false;
      }
    }
  }
  y[tid] = best;
  argmax[tid] = best_i;
}

__global__ void k_maxpool_bwd(const double* dy, const unsigned long long* argmax, double* dxp, size_t N, size_t C,
                              size_t Hp, size_t Wp, size_t Ho, size_t Wo) {
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  size_t nout = N * C * Ho * Wo;
  if (tid >= nout) return;
  unsigned long long a = argmax[tid];
  atomicAdd(dxp + a, dy[tid]);
}

void gpu_maxpool2d_forward_nchw(const double* x, double* y, unsigned long long* argmax, size_t N, size_t C, size_t H,
                                size_t W, size_t kH, size_t kW, size_t sh, size_t sw, size_t ph, size_t pw, size_t Ho,
                                size_t Wo, size_t Hp, size_t Wp) {
  double* xp = gpu_alloc(N * C * Hp * Wp);
  gpu_zero(xp, N * C * Hp * Wp);
  k_nchw_pad<<<grid1d(N * C * H * W), BLOCK_SIZE>>>(x, xp, N, C, H, W, ph, pw, Hp, Wp);
  KERNEL_CHECK();
  size_t nout = N * C * Ho * Wo;
  k_maxpool_fwd<<<grid1d(nout), BLOCK_SIZE>>>(xp, y, argmax, N, C, Hp, Wp, kH, kW, sh, sw, Ho, Wo);
  KERNEL_CHECK();
  gpu_free(xp);
}

void gpu_maxpool2d_backward_nchw(const double* dy, const unsigned long long* argmax, double* dx, size_t N, size_t C,
                                 size_t H, size_t W, size_t ph, size_t pw, size_t Ho, size_t Wo, size_t Hp, size_t Wp) {
  double* dxp = gpu_alloc(N * C * Hp * Wp);
  gpu_zero(dxp, N * C * Hp * Wp);
  size_t nout = N * C * Ho * Wo;
  k_maxpool_bwd<<<grid1d(nout), BLOCK_SIZE>>>(dy, argmax, dxp, N, C, Hp, Wp, Ho, Wo);
  KERNEL_CHECK();
  gpu_zero(dx, N * C * H * W);
  k_nchw_unpad<<<grid1d(N * C * H * W), BLOCK_SIZE>>>(dxp, dx, N, C, H, W, ph, pw, Hp, Wp);
  KERNEL_CHECK();
  gpu_free(dxp);
}

__global__ void k_avgpool_fwd(const double* xp, double* y, size_t N, size_t C, size_t Hp, size_t Wp, size_t kH,
                              size_t kW, size_t sh, size_t sw, size_t Ho, size_t Wo) {
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  size_t nout = N * C * Ho * Wo;
  if (tid >= nout) return;
  size_t ow = tid % Wo;
  size_t t = tid / Wo;
  size_t oh = t % Ho;
  t /= Ho;
  size_t c = t % C;
  size_t n = t / C;
  double s = 0.0;
  for (size_t kh = 0; kh < kH; ++kh) {
    for (size_t kw = 0; kw < kW; ++kw) {
      size_t ih = oh * sh + kh;
      size_t iw = ow * sw + kw;
      s += xp[((n * C + c) * Hp + ih) * Wp + iw];
    }
  }
  y[tid] = s / static_cast<double>(kH * kW);
}

__global__ void k_avgpool_bwd(const double* dy, double* dxp, size_t N, size_t C, size_t Hp, size_t Wp, size_t kH, size_t kW,
                              size_t sh, size_t sw, size_t Ho, size_t Wo, double scale) {
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  size_t nout = N * C * Ho * Wo;
  if (tid >= nout) return;
  size_t ow = tid % Wo;
  size_t t = tid / Wo;
  size_t oh = t % Ho;
  t /= Ho;
  size_t c = t % C;
  size_t n = t / C;
  double g = dy[tid] * scale;
  for (size_t kh = 0; kh < kH; ++kh) {
    for (size_t kw = 0; kw < kW; ++kw) {
      size_t ih = oh * sh + kh;
      size_t iw = ow * sw + kw;
      size_t ix = ((n * C + c) * Hp + ih) * Wp + iw;
      atomicAdd(dxp + ix, g);
    }
  }
}

void gpu_avgpool2d_forward_nchw(const double* x, double* y, size_t N, size_t C, size_t H, size_t W, size_t kH, size_t kW,
                                size_t sh, size_t sw, size_t ph, size_t pw, size_t Ho, size_t Wo, size_t Hp, size_t Wp) {
  double* xp = gpu_alloc(N * C * Hp * Wp);
  gpu_zero(xp, N * C * Hp * Wp);
  k_nchw_pad<<<grid1d(N * C * H * W), BLOCK_SIZE>>>(x, xp, N, C, H, W, ph, pw, Hp, Wp);
  KERNEL_CHECK();
  size_t nout = N * C * Ho * Wo;
  k_avgpool_fwd<<<grid1d(nout), BLOCK_SIZE>>>(xp, y, N, C, Hp, Wp, kH, kW, sh, sw, Ho, Wo);
  KERNEL_CHECK();
  gpu_free(xp);
}

void gpu_avgpool2d_backward_nchw(const double* dy, double* dx, size_t N, size_t C, size_t H, size_t W, size_t kH,
                                 size_t kW, size_t sh, size_t sw, size_t ph, size_t pw, size_t Ho, size_t Wo, size_t Hp,
                                 size_t Wp) {
  double* dxp = gpu_alloc(N * C * Hp * Wp);
  gpu_zero(dxp, N * C * Hp * Wp);
  double sc = 1.0 / static_cast<double>(kH * kW);
  size_t nout = N * C * Ho * Wo;
  k_avgpool_bwd<<<grid1d(nout), BLOCK_SIZE>>>(dy, dxp, N, C, Hp, Wp, kH, kW, sh, sw, Ho, Wo, sc);
  KERNEL_CHECK();
  gpu_zero(dx, N * C * H * W);
  k_nchw_unpad<<<grid1d(N * C * H * W), BLOCK_SIZE>>>(dxp, dx, N, C, H, W, ph, pw, Hp, Wp);
  KERNEL_CHECK();
  gpu_free(dxp);
}

__global__ void k_conv_tr_fwd(const double* x, const double* w, double* y, size_t N, size_t Ci, size_t Hi, size_t Wi,
                              size_t Co, size_t kH, size_t kW, size_t sh, size_t sw, size_t ph, size_t pw, size_t Ho,
                              size_t Wo) {
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  size_t total = N * Ci * Hi * Wi;
  if (tid >= total) return;
  size_t wi = tid % Wi;
  size_t t = tid / Wi;
  size_t hi = t % Hi;
  t /= Hi;
  size_t ci = t % Ci;
  size_t n = t / Ci;
  double xv = x[((n * Ci + ci) * Hi + hi) * Wi + wi];
  for (size_t co = 0; co < Co; ++co) {
    for (size_t kh_i = 0; kh_i < kH; ++kh_i) {
      for (size_t kw_i = 0; kw_i < kW; ++kw_i) {
        long long ho = static_cast<long long>(hi) * static_cast<long long>(sh) +
                         static_cast<long long>(kh_i) - static_cast<long long>(ph);
        long long wo = static_cast<long long>(wi) * static_cast<long long>(sw) +
                         static_cast<long long>(kw_i) - static_cast<long long>(pw);
        if (ho < 0 || wo < 0 || ho >= static_cast<long long>(Ho) || wo >= static_cast<long long>(Wo)) continue;
        size_t wi_idx = ((ci * Co + co) * kH + kh_i) * kW + kw_i;
        size_t yi = ((n * Co + co) * Ho + static_cast<size_t>(ho)) * Wo + static_cast<size_t>(wo);
        atomicAdd(y + yi, xv * w[wi_idx]);
      }
    }
  }
}

__global__ void k_conv_tr_bwd(const double* dy, const double* x, const double* w, double* dx, double* dw, size_t N,
                              size_t Ci, size_t Hi, size_t Wi, size_t Co, size_t kH, size_t kW, size_t sh, size_t sw,
                              size_t ph, size_t pw, size_t Ho, size_t Wo) {
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  size_t total = N * Ci * Hi * Wi;
  if (tid >= total) return;
  size_t wi = tid % Wi;
  size_t t = tid / Wi;
  size_t hi = t % Hi;
  t /= Hi;
  size_t ci = t % Ci;
  size_t n = t / Ci;
  size_t xi = ((n * Ci + ci) * Hi + hi) * Wi + wi;
  double xv = x[xi];
  for (size_t co = 0; co < Co; ++co) {
    for (size_t kh_i = 0; kh_i < kH; ++kh_i) {
      for (size_t kw_i = 0; kw_i < kW; ++kw_i) {
        long long ho = static_cast<long long>(hi) * static_cast<long long>(sh) +
                       static_cast<long long>(kh_i) - static_cast<long long>(ph);
        long long wo = static_cast<long long>(wi) * static_cast<long long>(sw) +
                       static_cast<long long>(kw_i) - static_cast<long long>(pw);
        if (ho < 0 || wo < 0 || ho >= static_cast<long long>(Ho) || wo >= static_cast<long long>(Wo)) continue;
        size_t wi_idx = ((ci * Co + co) * kH + kh_i) * kW + kw_i;
        size_t yi = ((n * Co + co) * Ho + static_cast<size_t>(ho)) * Wo + static_cast<size_t>(wo);
        double g = dy[yi];
        if (g == 0.0) continue;
        atomicAdd(dx + xi, g * w[wi_idx]);
        atomicAdd(dw + wi_idx, g * xv);
      }
    }
  }
}

void gpu_conv_transpose2d_forward_nchw(const double* x, const double* w, double* y, size_t N, size_t Ci, size_t Hi,
                                       size_t Wi, size_t Co, size_t kH, size_t kW, size_t sh, size_t sw, size_t ph,
                                       size_t pw, size_t Ho, size_t Wo) {
  size_t total_out = N * Co * Ho * Wo;
  gpu_zero(y, total_out);
  size_t total_in = N * Ci * Hi * Wi;
  k_conv_tr_fwd<<<grid1d(total_in), BLOCK_SIZE>>>(x, w, y, N, Ci, Hi, Wi, Co, kH, kW, sh, sw, ph, pw, Ho, Wo);
  KERNEL_CHECK();
}

void gpu_conv_transpose2d_backward_nchw(const double* dy, const double* x, const double* w, double* dx, double* dw,
                                        size_t N, size_t Ci, size_t Hi, size_t Wi, size_t Co, size_t kH, size_t kW,
                                        size_t sh, size_t sw, size_t ph, size_t pw, size_t Ho, size_t Wo) {
  gpu_zero(dx, N * Ci * Hi * Wi);
  gpu_zero(dw, Ci * Co * kH * kW);
  size_t total_in = N * Ci * Hi * Wi;
  k_conv_tr_bwd<<<grid1d(total_in), BLOCK_SIZE>>>(dy, x, w, dx, dw, N, Ci, Hi, Wi, Co, kH, kW, sh, sw, ph, pw, Ho,
                                                    Wo);
  KERNEL_CHECK();
}
