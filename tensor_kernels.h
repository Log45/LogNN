#pragma once
#include <cstddef>
#include <cstdint>
#include <string>

// Multi-backend ops: see BACKEND_KERNELS.md.

enum class DeviceType {
  CPU = 0,
  CUDA = 1,
  MLX = 2,
  MPS = 3,
};

struct Device {
  DeviceType type = DeviceType::CPU;
  int index = 0;

  Device() = default;
  Device(DeviceType type, int index) : type(type), index(index) {}
};

Device parse_device(const std::string& device_name, int index = 0);
const char* device_type_name(DeviceType type);
bool device_equal(const Device& a, const Device& b);

// Legacy CUDA kernels (implemented in tensor_kernels.cu)
double* gpu_alloc(size_t n);
void gpu_free(double* ptr);
void gpu_upload(double* dst, const double* src, size_t n);
void gpu_download(double* dst, const double* src, size_t n);
void gpu_copy_device(double* dst, const double* src, size_t n);
void gpu_zero(double* ptr, size_t n);
void gpu_sync();

// Element-wise operations
void gpu_neg(const double* a, double* out, size_t n);
void gpu_reciprocal(const double* a, double* out, size_t n);
void gpu_add_impl(const double* a, const double* b, double* out, size_t n);
void gpu_subtract(const double* a, const double* b, double* out, size_t n);
void gpu_mult_scalar(const double* a, double s, double* out, size_t n);
void gpu_elementwise_mult(const double* a, const double* b, double* out, size_t n);
void gpu_pow(const double* a, double p, double* out, size_t n);
void gpu_relu(const double* a, double* out, size_t n);
void gpu_binarilize(const double* a, double* out, size_t n);
void gpu_exp(const double* a, double* out, size_t n);
void gpu_sigmoid(const double* a, double* out, size_t n);
void gpu_tanh(const double* a, double* out, size_t n);
void gpu_softmax_last_dim(const double* a, double* out, size_t rows, size_t cols);
void gpu_sum_all(const double* a, double* out, size_t n);
void gpu_add_rowwise(const double* a, const double* row, double* out, size_t batch, size_t n);

// Transpose
void gpu_transpose_2d(const double* a, double* out, size_t rows, size_t cols);
void gpu_transpose_3d(const double* a, double* out, size_t B, size_t R, size_t C);

// Matmul (custom CUDA kernel, no cuBLAS)
void gpu_matmul_2d(const double* A, const double* B, double* C,
                   size_t M, size_t K, size_t N);
void gpu_matmul_batched(const double* A, const double* B, double* C,
                        size_t batch, size_t M, size_t K, size_t N);

// Device-agnostic backend interface used by Tensor.
double* backend_alloc(const Device& device, size_t n);
void backend_free(const Device& device, double* ptr);
void backend_upload(const Device& device, double* dst, const double* src, size_t n);
void backend_download(const Device& device, double* dst, const double* src, size_t n);
void backend_copy_device(const Device& device, double* dst, const double* src, size_t n);
void backend_zero(const Device& device, double* ptr, size_t n);
void backend_sync(const Device& device);

void backend_neg(const Device& device, const double* a, double* out, size_t n);
void backend_reciprocal(const Device& device, const double* a, double* out, size_t n);
void backend_add(const Device& device, const double* a, const double* b, double* out, size_t n);
void backend_subtract(const Device& device, const double* a, const double* b, double* out, size_t n);
void backend_mult_scalar(const Device& device, const double* a, double s, double* out, size_t n);
void backend_elementwise_mult(const Device& device, const double* a, const double* b, double* out, size_t n);
void backend_pow(const Device& device, const double* a, double p, double* out, size_t n);
void backend_relu(const Device& device, const double* a, double* out, size_t n);
void backend_binarilize(const Device& device, const double* a, double* out, size_t n);
void backend_exp(const Device& device, const double* a, double* out, size_t n);
void backend_sigmoid(const Device& device, const double* a, double* out, size_t n);
void backend_tanh(const Device& device, const double* a, double* out, size_t n);
void backend_softmax_last_dim(const Device& device, const double* a, double* out, size_t rows, size_t cols);
void backend_sum_all(const Device& device, const double* a, double* out, size_t n);
void backend_add_rowwise(const Device& device, const double* a, const double* row, double* out,
                         size_t batch, size_t n);

void backend_transpose_2d(const Device& device, const double* a, double* out, size_t rows, size_t cols);
void backend_transpose_3d(const Device& device, const double* a, double* out, size_t B, size_t R, size_t C);

void backend_matmul_2d(const Device& device, const double* A, const double* B, double* C,
                       size_t M, size_t K, size_t N);
void backend_matmul_batched(const Device& device, const double* A, const double* B, double* C,
                            size_t batch, size_t M, size_t K, size_t N);

// MLX/Apple backend runtime diagnostics.
bool backend_mlx_native_available();
size_t backend_mlx_dispatch_count();
void backend_mlx_reset_dispatch_count();
bool mlx_native_available();
size_t mlx_dispatch_count();
void mlx_reset_dispatch_count();

// MLX/Metal backend entrypoints (implemented in tensor_kernels_mlx.mm when WITH_MLX).
double* mlx_alloc(size_t n);
void mlx_free(double* ptr);
void mlx_upload(double* dst, const double* src, size_t n);
void mlx_download(double* dst, const double* src, size_t n);
void mlx_copy_device(double* dst, const double* src, size_t n);
void mlx_zero(double* ptr, size_t n);
void mlx_sync();
void mlx_neg(const double* a, double* out, size_t n);
void mlx_reciprocal(const double* a, double* out, size_t n);
void mlx_add(const double* a, const double* b, double* out, size_t n);
void mlx_subtract(const double* a, const double* b, double* out, size_t n);
void mlx_mult_scalar(const double* a, double s, double* out, size_t n);
void mlx_elementwise_mult(const double* a, const double* b, double* out, size_t n);
void mlx_pow(const double* a, double p, double* out, size_t n);
void mlx_relu(const double* a, double* out, size_t n);
void mlx_binarilize(const double* a, double* out, size_t n);
void mlx_exp(const double* a, double* out, size_t n);
void mlx_sigmoid(const double* a, double* out, size_t n);
void mlx_tanh(const double* a, double* out, size_t n);
void mlx_softmax_last_dim(const double* a, double* out, size_t rows, size_t cols);
void mlx_sum_all(const double* a, double* out, size_t n);
void mlx_add_rowwise(const double* a, const double* row, double* out, size_t batch, size_t n);
void mlx_transpose_2d(const double* a, double* out, size_t rows, size_t cols);
void mlx_transpose_3d(const double* a, double* out, size_t B, size_t R, size_t C);
void mlx_matmul_2d(const double* A, const double* B, double* C, size_t M, size_t K, size_t N);
void mlx_matmul_batched(const double* A, const double* B, double* C, size_t batch, size_t M, size_t K, size_t N);

#if defined(WITH_MLX)
uint32_t* mlx_alloc_u32(size_t n);
void mlx_free_u32(uint32_t* p);
void mlx_conv2d_forward_nchw(const double* x, const double* w, double* y, size_t N, size_t Ci, size_t H, size_t W,
                             size_t Co, size_t kH, size_t kW, size_t sh, size_t sw, size_t ph, size_t pw, size_t Ho,
                             size_t Wo);
void mlx_conv2d_backward_nchw(const double* dy, const double* x, const double* w, double* dx, double* dw, size_t N,
                              size_t Ci, size_t H, size_t W, size_t Co, size_t kH, size_t kW, size_t sh, size_t sw,
                              size_t ph, size_t pw, size_t Ho, size_t Wo);
void mlx_maxpool2d_forward_nchw(const double* x, double* y, uint32_t* argmax, size_t N, size_t C, size_t H, size_t W,
                                size_t kH, size_t kW, size_t sh, size_t sw, size_t ph, size_t pw, size_t Ho, size_t Wo,
                                size_t Hp, size_t Wp);
void mlx_maxpool2d_backward_nchw(const double* dy, const uint32_t* argmax, double* dx, size_t N, size_t C, size_t H,
                                 size_t W, size_t ph, size_t pw, size_t Ho, size_t Wo, size_t Hp, size_t Wp);
void mlx_avgpool2d_forward_nchw(const double* x, double* y, size_t N, size_t C, size_t H, size_t W, size_t kH,
                                size_t kW, size_t sh, size_t sw, size_t ph, size_t pw, size_t Ho, size_t Wo, size_t Hp,
                                size_t Wp);
void mlx_avgpool2d_backward_nchw(const double* dy, double* dx, size_t N, size_t C, size_t H, size_t W, size_t kH,
                                 size_t kW, size_t sh, size_t sw, size_t ph, size_t pw, size_t Ho, size_t Wo, size_t Hp,
                                 size_t Wp);
void mlx_conv_transpose2d_forward_nchw(const double* x, const double* w, double* y, size_t N, size_t Ci, size_t Hi,
                                       size_t Wi, size_t Co, size_t kH, size_t kW, size_t sh, size_t sw, size_t ph,
                                       size_t pw, size_t Ho, size_t Wo);
void mlx_conv_transpose2d_backward_nchw(const double* dy, const double* x, const double* w, double* dx, double* dw,
                                        size_t N, size_t Ci, size_t Hi, size_t Wi, size_t Co, size_t kH, size_t kW,
                                        size_t sh, size_t sw, size_t ph, size_t pw, size_t Ho, size_t Wo);
#endif

// --- Conv / pool (CUDA: tensor_kernels.cu; MLX: tensor_kernels_mlx.mm). ---
#if defined(WITH_CUDA)
unsigned long long* gpu_alloc_ull(size_t n);
void gpu_free_ull(unsigned long long* p);
void gpu_download_ull(const unsigned long long* src, unsigned long long* dst, size_t n);
void gpu_upload_ull(unsigned long long* dst, const unsigned long long* src, size_t n);
void gpu_conv2d_forward_nchw(const double* x, const double* w, double* y, size_t N, size_t Ci, size_t H, size_t W,
                             size_t Co, size_t kH, size_t kW, size_t sh, size_t sw, size_t ph, size_t pw, size_t Ho,
                             size_t Wo);
void gpu_conv2d_backward_nchw(const double* dy, const double* x, const double* w, double* dx, double* dw, size_t N,
                              size_t Ci, size_t H, size_t W, size_t Co, size_t kH, size_t kW, size_t sh, size_t sw,
                              size_t ph, size_t pw, size_t Ho, size_t Wo);
void gpu_maxpool2d_forward_nchw(const double* x, double* y, unsigned long long* argmax, size_t N, size_t C, size_t H,
                                size_t W, size_t kH, size_t kW, size_t sh, size_t sw, size_t ph, size_t pw, size_t Ho,
                                size_t Wo, size_t Hp, size_t Wp);
void gpu_maxpool2d_backward_nchw(const double* dy, const unsigned long long* argmax, double* dx, size_t N, size_t C,
                                 size_t H, size_t W, size_t ph, size_t pw, size_t Ho, size_t Wo, size_t Hp, size_t Wp);
void gpu_avgpool2d_forward_nchw(const double* x, double* y, size_t N, size_t C, size_t H, size_t W, size_t kH, size_t kW,
                                size_t sh, size_t sw, size_t ph, size_t pw, size_t Ho, size_t Wo, size_t Hp, size_t Wp);
void gpu_avgpool2d_backward_nchw(const double* dy, double* dx, size_t N, size_t C, size_t H, size_t W, size_t kH,
                                 size_t kW, size_t sh, size_t sw, size_t ph, size_t pw, size_t Ho, size_t Wo, size_t Hp,
                                 size_t Wp);
void gpu_conv_transpose2d_forward_nchw(const double* x, const double* w, double* y, size_t N, size_t Ci, size_t Hi,
                                       size_t Wi, size_t Co, size_t kH, size_t kW, size_t sh, size_t sw, size_t ph,
                                       size_t pw, size_t Ho, size_t Wo);
void gpu_conv_transpose2d_backward_nchw(const double* dy, const double* x, const double* w, double* dx, double* dw,
                                        size_t N, size_t Ci, size_t Hi, size_t Wi, size_t Co, size_t kH, size_t kW,
                                        size_t sh, size_t sw, size_t ph, size_t pw, size_t Ho, size_t Wo);
#endif
