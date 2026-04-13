#pragma once
#include <cstddef>

// GPU memory management
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

// Transpose
void gpu_transpose_2d(const double* a, double* out, size_t rows, size_t cols);
void gpu_transpose_3d(const double* a, double* out, size_t B, size_t R, size_t C);

// Matmul (custom CUDA kernel, no cuBLAS)
void gpu_matmul_2d(const double* A, const double* B, double* C,
                   size_t M, size_t K, size_t N);
void gpu_matmul_batched(const double* A, const double* B, double* C,
                        size_t batch, size_t M, size_t K, size_t N);
