#pragma once
#include <vector>
#include <stdexcept>
#include <cstdio>
#include <string>
#include "tensor_kernels.h"

class Tensor {
public:
  std::vector<size_t> dims;
  Device device;
  double* data;
  size_t total_size;
  bool owns_data;

  static size_t compute_size(const std::vector<size_t>& dims) {
    size_t len = 1;
    for (auto d : dims) len *= d;
    return len;
  }

  static void ensure_same_device(const Tensor& a, const Tensor& b, const char* op) {
    if (!device_equal(a.device, b.device)) {
      throw std::runtime_error(std::string("Device mismatch in ") + op);
    }
  }

  Tensor(std::vector<size_t> dims, std::string device_name = "cpu", int device_index = 0)
      : dims(dims), device(parse_device(device_name, device_index)), data(nullptr), total_size(0), owns_data(true) {
    total_size = compute_size(dims);
    data = backend_alloc(device, total_size);
    backend_zero(device, data, total_size);
  }

  Tensor(std::vector<size_t> dims, double* dev_ptr, size_t sz, Device dev)
      : dims(dims), device(dev), data(dev_ptr), total_size(sz), owns_data(true) {}

  Tensor(std::vector<size_t> dims, std::vector<std::vector<size_t>> idx,
         std::vector<double> val, std::string device_name = "cpu", int device_index = 0)
      : dims(dims), device(parse_device(device_name, device_index)), data(nullptr), total_size(0), owns_data(true) {
    total_size = compute_size(dims);
    std::vector<double> host_data(total_size, 0.0);
    if (idx.size() != val.size())
      throw std::runtime_error("Mismatched idx and val size");
    for (size_t i = 0; i < idx.size(); ++i)
      host_data[host_index(idx[i])] = val[i];
    data = backend_alloc(device, total_size);
    backend_upload(device, data, host_data.data(), total_size);
  }

  Tensor(const Tensor& other)
      : dims(other.dims), device(other.device), data(nullptr), total_size(other.total_size), owns_data(true) {
    data = backend_alloc(device, total_size);
    backend_copy_device(device, data, other.data, total_size);
  }

  Tensor(Tensor&& other) noexcept
      : dims(std::move(other.dims)), device(other.device), data(other.data),
        total_size(other.total_size), owns_data(other.owns_data) {
    other.data = nullptr;
    other.owns_data = false;
  }

  Tensor& operator=(const Tensor& other) {
    if (this != &other) {
      if (owns_data && data) backend_free(device, data);
      dims = other.dims;
      device = other.device;
      total_size = other.total_size;
      owns_data = true;
      data = backend_alloc(device, total_size);
      backend_copy_device(device, data, other.data, total_size);
    }
    return *this;
  }

  Tensor& operator=(Tensor&& other) noexcept {
    if (this != &other) {
      if (owns_data && data) backend_free(device, data);
      dims = std::move(other.dims);
      device = other.device;
      data = other.data;
      total_size = other.total_size;
      owns_data = other.owns_data;
      other.data = nullptr;
      other.owns_data = false;
    }
    return *this;
  }

  ~Tensor() {
    if (owns_data && data) {
      backend_free(device, data);
      data = nullptr;
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

  static Tensor ones(std::vector<size_t> dims, std::string device_name = "cpu", int device_index = 0) {
    size_t sz = compute_size(dims);
    std::vector<double> host(sz, 1.0);
    Device dev = parse_device(device_name, device_index);
    double* ptr = backend_alloc(dev, sz);
    backend_upload(dev, ptr, host.data(), sz);
    return Tensor(dims, ptr, sz, dev);
  }

  static Tensor from_data(std::vector<size_t> dims, const std::vector<double>& values,
                          std::string device_name = "cpu", int device_index = 0) {
    size_t sz = compute_size(dims);
    if (values.size() != sz) {
      throw std::runtime_error("Mismatched data size in from_data");
    }
    Device dev = parse_device(device_name, device_index);
    double* ptr = backend_alloc(dev, sz);
    backend_upload(dev, ptr, values.data(), sz);
    return Tensor(dims, ptr, sz, dev);
  }

  size_t index(std::vector<size_t> x) { return host_index(x); }

  Tensor reshape(std::vector<size_t> new_dims) {
    size_t len = compute_size(new_dims);
    if (len != total_size)
      throw std::runtime_error("Mismatched dims in reshape");
    double* ptr = backend_alloc(device, total_size);
    backend_copy_device(device, ptr, data, total_size);
    return Tensor(new_dims, ptr, total_size, device);
  }

  Tensor transpose() {
    if (dims.size() == 2) {
      double* ptr = backend_alloc(device, total_size);
      backend_transpose_2d(device, data, ptr, dims[0], dims[1]);
      return Tensor({dims[1], dims[0]}, ptr, total_size, device);
    } else if (dims.size() == 3) {
      double* ptr = backend_alloc(device, total_size);
      backend_transpose_3d(device, data, ptr, dims[0], dims[1], dims[2]);
      return Tensor({dims[0], dims[2], dims[1]}, ptr, total_size, device);
    }
    throw std::runtime_error("The tensor must be 2D or batched 2D tensors");
  }

  Tensor neg() {
    double* ptr = backend_alloc(device, total_size);
    backend_neg(device, data, ptr, total_size);
    return Tensor(dims, ptr, total_size, device);
  }

  Tensor reciprocal() {
    double* ptr = backend_alloc(device, total_size);
    backend_reciprocal(device, data, ptr, total_size);
    return Tensor(dims, ptr, total_size, device);
  }

  Tensor add(const Tensor& x) {
    ensure_same_device(*this, x, "add");
    if (dims != x.dims)
      throw std::runtime_error("Mismatched shape in add");
    double* ptr = backend_alloc(device, total_size);
    backend_add(device, data, x.data, ptr, total_size);
    return Tensor(dims, ptr, total_size, device);
  }

  Tensor subtract(const Tensor& x) {
    ensure_same_device(*this, x, "subtract");
    if (dims != x.dims)
      throw std::runtime_error("Mismatched shape in subtract");
    double* ptr = backend_alloc(device, total_size);
    backend_subtract(device, data, x.data, ptr, total_size);
    return Tensor(dims, ptr, total_size, device);
  }

  Tensor mult(double x) {
    double* ptr = backend_alloc(device, total_size);
    backend_mult_scalar(device, data, x, ptr, total_size);
    return Tensor(dims, ptr, total_size, device);
  }

  Tensor elementwise_mult(const Tensor& x) {
    ensure_same_device(*this, x, "elementwise_mult");
    if (dims != x.dims)
      throw std::runtime_error("Mismatched shape in elementwise_mult");
    double* ptr = backend_alloc(device, total_size);
    backend_elementwise_mult(device, data, x.data, ptr, total_size);
    return Tensor(dims, ptr, total_size, device);
  }

  Tensor pow(double x) {
    double* ptr = backend_alloc(device, total_size);
    backend_pow(device, data, x, ptr, total_size);
    return Tensor(dims, ptr, total_size, device);
  }

  Tensor relu() {
    double* ptr = backend_alloc(device, total_size);
    backend_relu(device, data, ptr, total_size);
    return Tensor(dims, ptr, total_size, device);
  }

  Tensor binarilize() {
    double* ptr = backend_alloc(device, total_size);
    backend_binarilize(device, data, ptr, total_size);
    return Tensor(dims, ptr, total_size, device);
  }

  Tensor exp() {
    double* ptr = backend_alloc(device, total_size);
    backend_exp(device, data, ptr, total_size);
    return Tensor(dims, ptr, total_size, device);
  }

  Tensor sigmoid() {
    double* ptr = backend_alloc(device, total_size);
    backend_sigmoid(device, data, ptr, total_size);
    return Tensor(dims, ptr, total_size, device);
  }

  Tensor tanh() {
    double* ptr = backend_alloc(device, total_size);
    backend_tanh(device, data, ptr, total_size);
    return Tensor(dims, ptr, total_size, device);
  }

  Tensor add_rowwise(const Tensor& row) {
    ensure_same_device(*this, row, "add_rowwise");
    if (dims.size() != 2 || row.dims.size() != 2 || row.dims[0] != 1 || row.dims[1] != dims[1]) {
      throw std::runtime_error("add_rowwise expects [B,N] + [1,N]");
    }
    double* ptr = backend_alloc(device, total_size);
    backend_add_rowwise(device, data, row.data, ptr, dims[0], dims[1]);
    return Tensor(dims, ptr, total_size, device);
  }

  Tensor sum() {
    double* ptr = backend_alloc(device, 1);
    backend_sum_all(device, data, ptr, total_size);
    return Tensor({1}, ptr, 1, device);
  }

  Tensor mean() {
    Tensor s = sum();
    return s.mult(1.0 / static_cast<double>(total_size));
  }

  Tensor softmax_last_dim() {
    if (dims.size() != 2) {
      throw std::runtime_error("softmax_last_dim expects 2D tensor");
    }
    double* ptr = backend_alloc(device, total_size);
    backend_softmax_last_dim(device, data, ptr, dims[0], dims[1]);
    return Tensor(dims, ptr, total_size, device);
  }

  Tensor matmul(const Tensor& x) {
    ensure_same_device(*this, x, "matmul");
    if (x.dims.size() != 2)
      throw std::runtime_error("The right operand of matmul must be 2D tensors");
    if (dims.size() != 2 && dims.size() != 3)
      throw std::runtime_error("The left operand of matmul must be 2D tensors or batched 2D tensors");
    if (dims[dims.size() - 1] != x.dims[0])
      throw std::runtime_error("Mismatched matmul matrix dimentions");

    if (dims.size() == 2) {
      size_t M = dims[0], K = dims[1], N = x.dims[1];
      size_t out_sz = M * N;
      double* ptr = backend_alloc(device, out_sz);
      backend_zero(device, ptr, out_sz);
      backend_matmul_2d(device, data, x.data, ptr, M, K, N);
      return Tensor({M, N}, ptr, out_sz, device);
    } else {
      size_t batch = dims[0], M = dims[1], K = dims[2], N = x.dims[1];
      size_t out_sz = batch * M * N;
      double* ptr = backend_alloc(device, out_sz);
      backend_zero(device, ptr, out_sz);
      backend_matmul_batched(device, data, x.data, ptr, batch, M, K, N);
      return Tensor({batch, M, N}, ptr, out_sz, device);
    }
  }

  void print() {
    backend_sync(device);
    std::vector<double> host(total_size);
    backend_download(device, host.data(), data, total_size);
    for (auto v : host)
      printf("%s\n", std::to_string(v).c_str());
  }

  std::vector<double> get_data() const {
    backend_sync(device);
    std::vector<double> host(total_size);
    backend_download(device, host.data(), data, total_size);
    return host;
  }

  std::vector<size_t> get_dims() const { return dims; }
  std::string get_device_type() const { return std::string(device_type_name(device.type)); }
  int get_device_index() const { return device.index; }
};
