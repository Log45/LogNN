#pragma once
#include <vector>
#include <stdexcept>
#include <cstdio>
#include <string>
#include "tensor_kernels.h"

class Tensor {
public:
  std::vector<size_t> dims;
  double* d_data;
  size_t total_size;
  bool owns_data;

  static size_t compute_size(const std::vector<size_t>& dims) {
    size_t len = 1;
    for (auto d : dims) len *= d;
    return len;
  }

  Tensor(std::vector<size_t> dims) : dims(dims), d_data(nullptr), total_size(0), owns_data(true) {
    total_size = compute_size(dims);
    d_data = gpu_alloc(total_size);
    gpu_zero(d_data, total_size);
  }

  Tensor(std::vector<size_t> dims, double* dev_ptr, size_t sz)
      : dims(dims), d_data(dev_ptr), total_size(sz), owns_data(true) {}

  Tensor(std::vector<size_t> dims, std::vector<std::vector<size_t>> idx,
         std::vector<double> val)
      : dims(dims), d_data(nullptr), total_size(0), owns_data(true) {
    total_size = compute_size(dims);
    std::vector<double> host_data(total_size, 0.0);
    if (idx.size() != val.size())
      throw std::runtime_error("Mismatched idx and val size");
    for (size_t i = 0; i < idx.size(); ++i)
      host_data[host_index(idx[i])] = val[i];
    d_data = gpu_alloc(total_size);
    gpu_upload(d_data, host_data.data(), total_size);
  }

  Tensor(const Tensor& other)
      : dims(other.dims), d_data(nullptr), total_size(other.total_size), owns_data(true) {
    d_data = gpu_alloc(total_size);
    gpu_copy_device(d_data, other.d_data, total_size);
  }

  Tensor(Tensor&& other) noexcept
      : dims(std::move(other.dims)), d_data(other.d_data),
        total_size(other.total_size), owns_data(other.owns_data) {
    other.d_data = nullptr;
    other.owns_data = false;
  }

  Tensor& operator=(const Tensor& other) {
    if (this != &other) {
      if (owns_data && d_data) gpu_free(d_data);
      dims = other.dims;
      total_size = other.total_size;
      owns_data = true;
      d_data = gpu_alloc(total_size);
      gpu_copy_device(d_data, other.d_data, total_size);
    }
    return *this;
  }

  Tensor& operator=(Tensor&& other) noexcept {
    if (this != &other) {
      if (owns_data && d_data) gpu_free(d_data);
      dims = std::move(other.dims);
      d_data = other.d_data;
      total_size = other.total_size;
      owns_data = other.owns_data;
      other.d_data = nullptr;
      other.owns_data = false;
    }
    return *this;
  }

  ~Tensor() {
    if (owns_data && d_data) {
      gpu_free(d_data);
      d_data = nullptr;
    }
  }

  size_t host_index(const std::vector<size_t>& x) const {
    if (x.size() != dims.size())
      throw std::runtime_error("Mismatched dims in index");
    size_t ret = 0, prod = 1;
    for (int i = (int)dims.size() - 1; i >= 0; --i) {
      if (x[i] >= dims[i])
        throw std::runtime_error("Index out of bound");
      ret += x[i] * prod;
      prod *= dims[i];
    }
    return ret;
  }

  // ========== Public API ==========

  static Tensor ones(std::vector<size_t> dims) {
    size_t sz = compute_size(dims);
    std::vector<double> host(sz, 1.0);
    double* ptr = gpu_alloc(sz);
    gpu_upload(ptr, host.data(), sz);
    return Tensor(dims, ptr, sz);
  }

  size_t index(std::vector<size_t> x) { return host_index(x); }

  Tensor reshape(std::vector<size_t> new_dims) {
    size_t len = compute_size(new_dims);
    if (len != total_size)
      throw std::runtime_error("Mismatched dims in reshape");
    double* ptr = gpu_alloc(total_size);
    gpu_copy_device(ptr, d_data, total_size);
    return Tensor(new_dims, ptr, total_size);
  }

  Tensor transpose() {
    if (dims.size() == 2) {
      double* ptr = gpu_alloc(total_size);
      gpu_transpose_2d(d_data, ptr, dims[0], dims[1]);
      return Tensor({dims[1], dims[0]}, ptr, total_size);
    } else if (dims.size() == 3) {
      double* ptr = gpu_alloc(total_size);
      gpu_transpose_3d(d_data, ptr, dims[0], dims[1], dims[2]);
      return Tensor({dims[0], dims[2], dims[1]}, ptr, total_size);
    }
    throw std::runtime_error("The tensor must be 2D or batched 2D tensors");
  }

  Tensor neg() {
    double* ptr = gpu_alloc(total_size);
    gpu_neg(d_data, ptr, total_size);
    return Tensor(dims, ptr, total_size);
  }

  Tensor reciprocal() {
    double* ptr = gpu_alloc(total_size);
    gpu_reciprocal(d_data, ptr, total_size);
    return Tensor(dims, ptr, total_size);
  }

  Tensor add(const Tensor& x) {
    if (dims != x.dims)
      throw std::runtime_error("Mismatched shape in add");
    double* ptr = gpu_alloc(total_size);
    gpu_add_impl(d_data, x.d_data, ptr, total_size);
    return Tensor(dims, ptr, total_size);
  }

  Tensor subtract(const Tensor& x) {
    if (dims != x.dims)
      throw std::runtime_error("Mismatched shape in subtract");
    double* ptr = gpu_alloc(total_size);
    gpu_subtract(d_data, x.d_data, ptr, total_size);
    return Tensor(dims, ptr, total_size);
  }

  Tensor mult(double x) {
    double* ptr = gpu_alloc(total_size);
    gpu_mult_scalar(d_data, x, ptr, total_size);
    return Tensor(dims, ptr, total_size);
  }

  Tensor elementwise_mult(const Tensor& x) {
    if (dims != x.dims)
      throw std::runtime_error("Mismatched shape in elementwise_mult");
    double* ptr = gpu_alloc(total_size);
    gpu_elementwise_mult(d_data, x.d_data, ptr, total_size);
    return Tensor(dims, ptr, total_size);
  }

  Tensor pow(double x) {
    double* ptr = gpu_alloc(total_size);
    gpu_pow(d_data, x, ptr, total_size);
    return Tensor(dims, ptr, total_size);
  }

  Tensor relu() {
    double* ptr = gpu_alloc(total_size);
    gpu_relu(d_data, ptr, total_size);
    return Tensor(dims, ptr, total_size);
  }

  Tensor binarilize() {
    double* ptr = gpu_alloc(total_size);
    gpu_binarilize(d_data, ptr, total_size);
    return Tensor(dims, ptr, total_size);
  }

  Tensor exp() {
    double* ptr = gpu_alloc(total_size);
    gpu_exp(d_data, ptr, total_size);
    return Tensor(dims, ptr, total_size);
  }

  Tensor matmul(const Tensor& x) {
    if (x.dims.size() != 2)
      throw std::runtime_error("The right operand of matmul must be 2D tensors");
    if (dims.size() != 2 && dims.size() != 3)
      throw std::runtime_error("The left operand of matmul must be 2D tensors or batched 2D tensors");
    if (dims[dims.size() - 1] != x.dims[0])
      throw std::runtime_error("Mismatched matmul matrix dimentions");

    if (dims.size() == 2) {
      size_t M = dims[0], K = dims[1], N = x.dims[1];
      size_t out_sz = M * N;
      double* ptr = gpu_alloc(out_sz);
      gpu_zero(ptr, out_sz);
      gpu_matmul_2d(d_data, x.d_data, ptr, M, K, N);
      return Tensor({M, N}, ptr, out_sz);
    } else {
      size_t batch = dims[0], M = dims[1], K = dims[2], N = x.dims[1];
      size_t out_sz = batch * M * N;
      double* ptr = gpu_alloc(out_sz);
      gpu_zero(ptr, out_sz);
      gpu_matmul_batched(d_data, x.d_data, ptr, batch, M, K, N);
      return Tensor({batch, M, N}, ptr, out_sz);
    }
  }

  void print() {
    gpu_sync();
    std::vector<double> host(total_size);
    gpu_download(host.data(), d_data, total_size);
    for (auto v : host)
      printf("%s\n", std::to_string(v).c_str());
  }

  std::vector<double> get_data() {
    gpu_sync();
    std::vector<double> host(total_size);
    gpu_download(host.data(), d_data, total_size);
    return host;
  }

  std::vector<size_t> get_dims() { return dims; }
};
