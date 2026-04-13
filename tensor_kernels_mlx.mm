#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <atomic>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "tensor_kernels.h"

namespace {

const char* kMetalSource = R"METAL(
#include <metal_stdlib>
using namespace metal;

kernel void k_neg(const device float* a [[buffer(0)]], device float* out [[buffer(1)]], constant uint& n [[buffer(2)]], uint gid [[thread_position_in_grid]]) { if (gid < n) out[gid] = -a[gid]; }
kernel void k_reciprocal(const device float* a [[buffer(0)]], device float* out [[buffer(1)]], constant uint& n [[buffer(2)]], uint gid [[thread_position_in_grid]]) { if (gid < n) out[gid] = 1.0f / a[gid]; }
kernel void k_add(const device float* a [[buffer(0)]], const device float* b [[buffer(1)]], device float* out [[buffer(2)]], constant uint& n [[buffer(3)]], uint gid [[thread_position_in_grid]]) { if (gid < n) out[gid] = a[gid] + b[gid]; }
kernel void k_subtract(const device float* a [[buffer(0)]], const device float* b [[buffer(1)]], device float* out [[buffer(2)]], constant uint& n [[buffer(3)]], uint gid [[thread_position_in_grid]]) { if (gid < n) out[gid] = a[gid] - b[gid]; }
kernel void k_mult_scalar(const device float* a [[buffer(0)]], device float* out [[buffer(1)]], constant float& s [[buffer(2)]], constant uint& n [[buffer(3)]], uint gid [[thread_position_in_grid]]) { if (gid < n) out[gid] = a[gid] * s; }
kernel void k_elementwise_mult(const device float* a [[buffer(0)]], const device float* b [[buffer(1)]], device float* out [[buffer(2)]], constant uint& n [[buffer(3)]], uint gid [[thread_position_in_grid]]) { if (gid < n) out[gid] = a[gid] * b[gid]; }
kernel void k_pow(const device float* a [[buffer(0)]], device float* out [[buffer(1)]], constant float& p [[buffer(2)]], constant uint& n [[buffer(3)]], uint gid [[thread_position_in_grid]]) { if (gid < n) out[gid] = pow(a[gid], p); }
kernel void k_relu(const device float* a [[buffer(0)]], device float* out [[buffer(1)]], constant uint& n [[buffer(2)]], uint gid [[thread_position_in_grid]]) { if (gid < n) out[gid] = a[gid] > 0.0f ? a[gid] : 0.0f; }
kernel void k_binarilize(const device float* a [[buffer(0)]], device float* out [[buffer(1)]], constant uint& n [[buffer(2)]], uint gid [[thread_position_in_grid]]) { if (gid < n) out[gid] = a[gid] > 0.0f ? 1.0f : 0.0f; }
kernel void k_exp(const device float* a [[buffer(0)]], device float* out [[buffer(1)]], constant uint& n [[buffer(2)]], uint gid [[thread_position_in_grid]]) { if (gid < n) out[gid] = exp(a[gid]); }
kernel void k_sigmoid(const device float* a [[buffer(0)]], device float* out [[buffer(1)]], constant uint& n [[buffer(2)]], uint gid [[thread_position_in_grid]]) { if (gid < n) out[gid] = 1.0f / (1.0f + exp(-a[gid])); }
kernel void k_tanh(const device float* a [[buffer(0)]], device float* out [[buffer(1)]], constant uint& n [[buffer(2)]], uint gid [[thread_position_in_grid]]) { if (gid < n) out[gid] = tanh(a[gid]); }
kernel void k_add_rowwise(const device float* a [[buffer(0)]], const device float* row [[buffer(1)]], device float* out [[buffer(2)]], constant uint& batch [[buffer(3)]], constant uint& n [[buffer(4)]], uint gid [[thread_position_in_grid]]) {
  uint total = batch * n;
  if (gid < total) {
    uint j = gid % n;
    out[gid] = a[gid] + row[j];
  }
}
kernel void k_transpose2d(const device float* a [[buffer(0)]], device float* out [[buffer(1)]], constant uint& rows [[buffer(2)]], constant uint& cols [[buffer(3)]], uint2 gid [[thread_position_in_grid]]) {
  if (gid.x < cols && gid.y < rows) out[gid.x * rows + gid.y] = a[gid.y * cols + gid.x];
}
kernel void k_transpose3d(const device float* a [[buffer(0)]], device float* out [[buffer(1)]], constant uint& B [[buffer(2)]], constant uint& R [[buffer(3)]], constant uint& C [[buffer(4)]], uint gid [[thread_position_in_grid]]) {
  uint total = B * R * C;
  if (gid < total) {
    uint b = gid / (R * C);
    uint rem = gid % (R * C);
    uint i = rem / C;
    uint j = rem % C;
    out[b * C * R + j * R + i] = a[gid];
  }
}
kernel void k_matmul2d(const device float* A [[buffer(0)]], const device float* B [[buffer(1)]], device float* C [[buffer(2)]], constant uint& M [[buffer(3)]], constant uint& K [[buffer(4)]], constant uint& N [[buffer(5)]], uint2 gid [[thread_position_in_grid]]) {
  if (gid.x < N && gid.y < M) {
    float sum = 0.0f;
    for (uint k = 0; k < K; ++k) sum += A[gid.y * K + k] * B[k * N + gid.x];
    C[gid.y * N + gid.x] += sum;
  }
}
kernel void k_matmul_batched(const device float* A [[buffer(0)]], const device float* B [[buffer(1)]], device float* C [[buffer(2)]], constant uint& batch [[buffer(3)]], constant uint& M [[buffer(4)]], constant uint& K [[buffer(5)]], constant uint& N [[buffer(6)]], uint gid [[thread_position_in_grid]]) {
  uint total = batch * M * N;
  if (gid < total) {
    uint b = gid / (M * N);
    uint rem = gid % (M * N);
    uint i = rem / N;
    uint j = rem % N;
    float sum = 0.0f;
    const device float* A_b = A + b * M * K;
    device float* C_b = C + b * M * N;
    for (uint k = 0; k < K; ++k) sum += A_b[i * K + k] * B[k * N + j];
    C_b[i * N + j] += sum;
  }
}
kernel void k_softmax_last_dim(const device float* a [[buffer(0)]], device float* out [[buffer(1)]], constant uint& rows [[buffer(2)]], constant uint& cols [[buffer(3)]], uint gid [[thread_position_in_grid]]) {
  if (gid < rows) {
    uint base = gid * cols;
    float mx = a[base];
    for (uint j = 1; j < cols; ++j) mx = max(mx, a[base + j]);
    float sum = 0.0f;
    for (uint j = 0; j < cols; ++j) {
      float e = exp(a[base + j] - mx);
      out[base + j] = e;
      sum += e;
    }
    float inv = 1.0f / sum;
    for (uint j = 0; j < cols; ++j) out[base + j] *= inv;
  }
}
kernel void k_sum_all(const device float* a [[buffer(0)]], device float* out [[buffer(1)]], constant uint& n [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
  if (gid == 0) {
    float s = 0.0f;
    for (uint i = 0; i < n; ++i) s += a[i];
    out[0] = s;
  }
}
)METAL";

std::atomic<size_t> g_dispatch_count{0};

struct BufferRecord {
  id<MTLBuffer> buffer = nil;
  size_t n = 0;
};

class AppleMlxContext {
 public:
  static AppleMlxContext& instance() {
    static AppleMlxContext ctx;
    return ctx;
  }

  bool available() const { return ok_; }

  id<MTLBuffer> get_buffer(double* token, size_t min_n = 0) {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = buffers_.find(ptr_to_key(token));
    if (it == buffers_.end()) throw std::runtime_error("Invalid MLX buffer token");
    if (min_n > 0 && it->second.n < min_n) throw std::runtime_error("MLX buffer too small");
    return it->second.buffer;
  }

  size_t get_size(double* token) {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = buffers_.find(ptr_to_key(token));
    if (it == buffers_.end()) throw std::runtime_error("Invalid MLX buffer token");
    return it->second.n;
  }

  double* alloc(size_t n) {
    ensure_ready();
    id<MTLBuffer> b = [device_ newBufferWithLength:n * sizeof(float) options:MTLResourceStorageModeShared];
    if (!b) throw std::runtime_error("Failed to allocate MLX Metal buffer");
    const uintptr_t key = next_key_.fetch_add(1);
    {
      std::lock_guard<std::mutex> lock(mu_);
      buffers_[key] = BufferRecord{b, n};
    }
    return key_to_ptr(key);
  }

  void free_buf(double* token) {
    std::lock_guard<std::mutex> lock(mu_);
    buffers_.erase(ptr_to_key(token));
  }

  id<MTLComputePipelineState> pipe(const char* name) {
    ensure_ready();
    std::string key(name);
    auto it = pipelines_.find(key);
    if (it != pipelines_.end()) return it->second;
    id<MTLFunction> fn = [library_ newFunctionWithName:[NSString stringWithUTF8String:name]];
    if (!fn) throw std::runtime_error(std::string("Metal function not found: ") + name);
    NSError* err = nil;
    id<MTLComputePipelineState> p = [device_ newComputePipelineStateWithFunction:fn error:&err];
    if (!p) throw std::runtime_error(std::string("Failed to create pipeline: ") + (err ? err.localizedDescription.UTF8String : ""));
    pipelines_[key] = p;
    return p;
  }

  id<MTLCommandBuffer> command_buffer() {
    ensure_ready();
    id<MTLCommandBuffer> cb = [queue_ commandBuffer];
    if (!cb) throw std::runtime_error("Failed to create command buffer");
    return cb;
  }

 private:
  AppleMlxContext() {
    device_ = MTLCreateSystemDefaultDevice();
    if (!device_) return;
    queue_ = [device_ newCommandQueue];
    if (!queue_) return;
    NSError* err = nil;
    NSString* src = [NSString stringWithUTF8String:kMetalSource];
    library_ = [device_ newLibraryWithSource:src options:nil error:&err];
    if (!library_) return;
    ok_ = true;
  }

  void ensure_ready() {
    if (!ok_) throw std::runtime_error("MLX/Metal backend unavailable");
  }

  static uintptr_t ptr_to_key(double* p) { return reinterpret_cast<uintptr_t>(p); }
  static double* key_to_ptr(uintptr_t k) { return reinterpret_cast<double*>(k); }

  bool ok_ = false;
  id<MTLDevice> device_ = nil;
  id<MTLCommandQueue> queue_ = nil;
  id<MTLLibrary> library_ = nil;
  std::unordered_map<std::string, id<MTLComputePipelineState>> pipelines_;
  std::unordered_map<uintptr_t, BufferRecord> buffers_;
  std::mutex mu_;
  std::atomic<uintptr_t> next_key_{1};
};

void run_1d(const char* kernel_name, uint32_t n, void (^bind)(id<MTLComputeCommandEncoder>)) {
  auto& ctx = AppleMlxContext::instance();
  id<MTLCommandBuffer> cb = ctx.command_buffer();
  id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
  id<MTLComputePipelineState> p = ctx.pipe(kernel_name);
  [enc setComputePipelineState:p];
  bind(enc);
  const NSUInteger tg = p.maxTotalThreadsPerThreadgroup > 0 ? p.maxTotalThreadsPerThreadgroup : 256;
  [enc dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
  [enc endEncoding];
  [cb commit];
  [cb waitUntilCompleted];
  g_dispatch_count.fetch_add(1);
}

void run_2d(const char* kernel_name, uint32_t x, uint32_t y, void (^bind)(id<MTLComputeCommandEncoder>)) {
  auto& ctx = AppleMlxContext::instance();
  id<MTLCommandBuffer> cb = ctx.command_buffer();
  id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
  id<MTLComputePipelineState> p = ctx.pipe(kernel_name);
  [enc setComputePipelineState:p];
  bind(enc);
  const NSUInteger tx = 16;
  const NSUInteger ty = 16;
  [enc dispatchThreads:MTLSizeMake(x, y, 1) threadsPerThreadgroup:MTLSizeMake(tx, ty, 1)];
  [enc endEncoding];
  [cb commit];
  [cb waitUntilCompleted];
  g_dispatch_count.fetch_add(1);
}

void copy_doubles_to_float_buffer(id<MTLBuffer> buf, const double* src, size_t n) {
  float* dst = static_cast<float*>([buf contents]);
  for (size_t i = 0; i < n; ++i) dst[i] = static_cast<float>(src[i]);
}

void copy_float_buffer_to_doubles(double* dst, id<MTLBuffer> buf, size_t n) {
  const float* src = static_cast<const float*>([buf contents]);
  for (size_t i = 0; i < n; ++i) dst[i] = static_cast<double>(src[i]);
}

}  // namespace

bool mlx_native_available() {
  return AppleMlxContext::instance().available();
}

size_t mlx_dispatch_count() {
  return g_dispatch_count.load();
}

void mlx_reset_dispatch_count() {
  g_dispatch_count.store(0);
}

double* mlx_alloc(size_t n) {
  return AppleMlxContext::instance().alloc(n);
}

void mlx_free(double* ptr) {
  AppleMlxContext::instance().free_buf(ptr);
}

void mlx_upload(double* dst, const double* src, size_t n) {
  auto& ctx = AppleMlxContext::instance();
  id<MTLBuffer> b = ctx.get_buffer(dst, n);
  copy_doubles_to_float_buffer(b, src, n);
}

void mlx_download(double* dst, const double* src, size_t n) {
  auto& ctx = AppleMlxContext::instance();
  id<MTLBuffer> b = ctx.get_buffer(const_cast<double*>(src), n);
  copy_float_buffer_to_doubles(dst, b, n);
}

void mlx_copy_device(double* dst, const double* src, size_t n) {
  auto& ctx = AppleMlxContext::instance();
  id<MTLBuffer> d = ctx.get_buffer(dst, n);
  id<MTLBuffer> s = ctx.get_buffer(const_cast<double*>(src), n);
  std::memcpy([d contents], [s contents], n * sizeof(float));
}

void mlx_zero(double* ptr, size_t n) {
  auto& ctx = AppleMlxContext::instance();
  id<MTLBuffer> b = ctx.get_buffer(ptr, n);
  std::memset([b contents], 0, n * sizeof(float));
}

void mlx_sync() {}

#define MLX_UNARY_IMPL(name, kernel) \
  void name(const double* a, double* out, size_t n) { \
    auto& ctx = AppleMlxContext::instance(); \
    id<MTLBuffer> a_b = ctx.get_buffer(const_cast<double*>(a), n); \
    id<MTLBuffer> o_b = ctx.get_buffer(out, n); \
    uint32_t nn = static_cast<uint32_t>(n); \
    run_1d(kernel, nn, ^(id<MTLComputeCommandEncoder> enc) { \
      [enc setBuffer:a_b offset:0 atIndex:0]; \
      [enc setBuffer:o_b offset:0 atIndex:1]; \
      [enc setBytes:&nn length:sizeof(nn) atIndex:2]; \
    }); \
  }

MLX_UNARY_IMPL(mlx_neg, "k_neg")
MLX_UNARY_IMPL(mlx_reciprocal, "k_reciprocal")
MLX_UNARY_IMPL(mlx_relu, "k_relu")
MLX_UNARY_IMPL(mlx_binarilize, "k_binarilize")
MLX_UNARY_IMPL(mlx_exp, "k_exp")
MLX_UNARY_IMPL(mlx_sigmoid, "k_sigmoid")
MLX_UNARY_IMPL(mlx_tanh, "k_tanh")

void mlx_add(const double* a, const double* b, double* out, size_t n) {
  auto& ctx = AppleMlxContext::instance();
  id<MTLBuffer> a_b = ctx.get_buffer(const_cast<double*>(a), n);
  id<MTLBuffer> b_b = ctx.get_buffer(const_cast<double*>(b), n);
  id<MTLBuffer> o_b = ctx.get_buffer(out, n);
  uint32_t nn = static_cast<uint32_t>(n);
  run_1d("k_add", nn, ^(id<MTLComputeCommandEncoder> enc) {
    [enc setBuffer:a_b offset:0 atIndex:0];
    [enc setBuffer:b_b offset:0 atIndex:1];
    [enc setBuffer:o_b offset:0 atIndex:2];
    [enc setBytes:&nn length:sizeof(nn) atIndex:3];
  });
}

void mlx_subtract(const double* a, const double* b, double* out, size_t n) {
  auto& ctx = AppleMlxContext::instance();
  id<MTLBuffer> a_b = ctx.get_buffer(const_cast<double*>(a), n);
  id<MTLBuffer> b_b = ctx.get_buffer(const_cast<double*>(b), n);
  id<MTLBuffer> o_b = ctx.get_buffer(out, n);
  uint32_t nn = static_cast<uint32_t>(n);
  run_1d("k_subtract", nn, ^(id<MTLComputeCommandEncoder> enc) {
    [enc setBuffer:a_b offset:0 atIndex:0];
    [enc setBuffer:b_b offset:0 atIndex:1];
    [enc setBuffer:o_b offset:0 atIndex:2];
    [enc setBytes:&nn length:sizeof(nn) atIndex:3];
  });
}

void mlx_mult_scalar(const double* a, double s, double* out, size_t n) {
  auto& ctx = AppleMlxContext::instance();
  id<MTLBuffer> a_b = ctx.get_buffer(const_cast<double*>(a), n);
  id<MTLBuffer> o_b = ctx.get_buffer(out, n);
  float sf = static_cast<float>(s);
  uint32_t nn = static_cast<uint32_t>(n);
  run_1d("k_mult_scalar", nn, ^(id<MTLComputeCommandEncoder> enc) {
    [enc setBuffer:a_b offset:0 atIndex:0];
    [enc setBuffer:o_b offset:0 atIndex:1];
    [enc setBytes:&sf length:sizeof(sf) atIndex:2];
    [enc setBytes:&nn length:sizeof(nn) atIndex:3];
  });
}

void mlx_elementwise_mult(const double* a, const double* b, double* out, size_t n) {
  auto& ctx = AppleMlxContext::instance();
  id<MTLBuffer> a_b = ctx.get_buffer(const_cast<double*>(a), n);
  id<MTLBuffer> b_b = ctx.get_buffer(const_cast<double*>(b), n);
  id<MTLBuffer> o_b = ctx.get_buffer(out, n);
  uint32_t nn = static_cast<uint32_t>(n);
  run_1d("k_elementwise_mult", nn, ^(id<MTLComputeCommandEncoder> enc) {
    [enc setBuffer:a_b offset:0 atIndex:0];
    [enc setBuffer:b_b offset:0 atIndex:1];
    [enc setBuffer:o_b offset:0 atIndex:2];
    [enc setBytes:&nn length:sizeof(nn) atIndex:3];
  });
}

void mlx_pow(const double* a, double p, double* out, size_t n) {
  auto& ctx = AppleMlxContext::instance();
  id<MTLBuffer> a_b = ctx.get_buffer(const_cast<double*>(a), n);
  id<MTLBuffer> o_b = ctx.get_buffer(out, n);
  float pf = static_cast<float>(p);
  uint32_t nn = static_cast<uint32_t>(n);
  run_1d("k_pow", nn, ^(id<MTLComputeCommandEncoder> enc) {
    [enc setBuffer:a_b offset:0 atIndex:0];
    [enc setBuffer:o_b offset:0 atIndex:1];
    [enc setBytes:&pf length:sizeof(pf) atIndex:2];
    [enc setBytes:&nn length:sizeof(nn) atIndex:3];
  });
}

void mlx_softmax_last_dim(const double* a, double* out, size_t rows, size_t cols) {
  auto& ctx = AppleMlxContext::instance();
  id<MTLBuffer> a_b = ctx.get_buffer(const_cast<double*>(a), rows * cols);
  id<MTLBuffer> o_b = ctx.get_buffer(out, rows * cols);
  uint32_t rr = static_cast<uint32_t>(rows);
  uint32_t cc = static_cast<uint32_t>(cols);
  run_1d("k_softmax_last_dim", rr, ^(id<MTLComputeCommandEncoder> enc) {
    [enc setBuffer:a_b offset:0 atIndex:0];
    [enc setBuffer:o_b offset:0 atIndex:1];
    [enc setBytes:&rr length:sizeof(rr) atIndex:2];
    [enc setBytes:&cc length:sizeof(cc) atIndex:3];
  });
}

void mlx_sum_all(const double* a, double* out, size_t n) {
  auto& ctx = AppleMlxContext::instance();
  id<MTLBuffer> a_b = ctx.get_buffer(const_cast<double*>(a), n);
  id<MTLBuffer> o_b = ctx.get_buffer(out, 1);
  uint32_t nn = static_cast<uint32_t>(n);
  run_1d("k_sum_all", 1, ^(id<MTLComputeCommandEncoder> enc) {
    [enc setBuffer:a_b offset:0 atIndex:0];
    [enc setBuffer:o_b offset:0 atIndex:1];
    [enc setBytes:&nn length:sizeof(nn) atIndex:2];
  });
}

void mlx_add_rowwise(const double* a, const double* row, double* out, size_t batch, size_t n) {
  auto& ctx = AppleMlxContext::instance();
  id<MTLBuffer> a_b = ctx.get_buffer(const_cast<double*>(a), batch * n);
  id<MTLBuffer> r_b = ctx.get_buffer(const_cast<double*>(row), n);
  id<MTLBuffer> o_b = ctx.get_buffer(out, batch * n);
  uint32_t bb = static_cast<uint32_t>(batch);
  uint32_t nn = static_cast<uint32_t>(n);
  run_1d("k_add_rowwise", bb * nn, ^(id<MTLComputeCommandEncoder> enc) {
    [enc setBuffer:a_b offset:0 atIndex:0];
    [enc setBuffer:r_b offset:0 atIndex:1];
    [enc setBuffer:o_b offset:0 atIndex:2];
    [enc setBytes:&bb length:sizeof(bb) atIndex:3];
    [enc setBytes:&nn length:sizeof(nn) atIndex:4];
  });
}

void mlx_transpose_2d(const double* a, double* out, size_t rows, size_t cols) {
  auto& ctx = AppleMlxContext::instance();
  id<MTLBuffer> a_b = ctx.get_buffer(const_cast<double*>(a), rows * cols);
  id<MTLBuffer> o_b = ctx.get_buffer(out, rows * cols);
  uint32_t rr = static_cast<uint32_t>(rows);
  uint32_t cc = static_cast<uint32_t>(cols);
  run_2d("k_transpose2d", cc, rr, ^(id<MTLComputeCommandEncoder> enc) {
    [enc setBuffer:a_b offset:0 atIndex:0];
    [enc setBuffer:o_b offset:0 atIndex:1];
    [enc setBytes:&rr length:sizeof(rr) atIndex:2];
    [enc setBytes:&cc length:sizeof(cc) atIndex:3];
  });
}

void mlx_transpose_3d(const double* a, double* out, size_t B, size_t R, size_t C) {
  auto& ctx = AppleMlxContext::instance();
  id<MTLBuffer> a_b = ctx.get_buffer(const_cast<double*>(a), B * R * C);
  id<MTLBuffer> o_b = ctx.get_buffer(out, B * R * C);
  uint32_t bb = static_cast<uint32_t>(B);
  uint32_t rr = static_cast<uint32_t>(R);
  uint32_t cc = static_cast<uint32_t>(C);
  run_1d("k_transpose3d", bb * rr * cc, ^(id<MTLComputeCommandEncoder> enc) {
    [enc setBuffer:a_b offset:0 atIndex:0];
    [enc setBuffer:o_b offset:0 atIndex:1];
    [enc setBytes:&bb length:sizeof(bb) atIndex:2];
    [enc setBytes:&rr length:sizeof(rr) atIndex:3];
    [enc setBytes:&cc length:sizeof(cc) atIndex:4];
  });
}

void mlx_matmul_2d(const double* A, const double* B, double* C, size_t M, size_t K, size_t N) {
  auto& ctx = AppleMlxContext::instance();
  id<MTLBuffer> a_b = ctx.get_buffer(const_cast<double*>(A), M * K);
  id<MTLBuffer> b_b = ctx.get_buffer(const_cast<double*>(B), K * N);
  id<MTLBuffer> c_b = ctx.get_buffer(C, M * N);
  uint32_t mm = static_cast<uint32_t>(M);
  uint32_t kk = static_cast<uint32_t>(K);
  uint32_t nn = static_cast<uint32_t>(N);
  run_2d("k_matmul2d", nn, mm, ^(id<MTLComputeCommandEncoder> enc) {
    [enc setBuffer:a_b offset:0 atIndex:0];
    [enc setBuffer:b_b offset:0 atIndex:1];
    [enc setBuffer:c_b offset:0 atIndex:2];
    [enc setBytes:&mm length:sizeof(mm) atIndex:3];
    [enc setBytes:&kk length:sizeof(kk) atIndex:4];
    [enc setBytes:&nn length:sizeof(nn) atIndex:5];
  });
}

void mlx_matmul_batched(const double* A, const double* B, double* C, size_t batch, size_t M, size_t K, size_t N) {
  auto& ctx = AppleMlxContext::instance();
  id<MTLBuffer> a_b = ctx.get_buffer(const_cast<double*>(A), batch * M * K);
  id<MTLBuffer> b_b = ctx.get_buffer(const_cast<double*>(B), K * N);
  id<MTLBuffer> c_b = ctx.get_buffer(C, batch * M * N);
  uint32_t bb = static_cast<uint32_t>(batch);
  uint32_t mm = static_cast<uint32_t>(M);
  uint32_t kk = static_cast<uint32_t>(K);
  uint32_t nn = static_cast<uint32_t>(N);
  run_1d("k_matmul_batched", bb * mm * nn, ^(id<MTLComputeCommandEncoder> enc) {
    [enc setBuffer:a_b offset:0 atIndex:0];
    [enc setBuffer:b_b offset:0 atIndex:1];
    [enc setBuffer:c_b offset:0 atIndex:2];
    [enc setBytes:&bb length:sizeof(bb) atIndex:3];
    [enc setBytes:&mm length:sizeof(mm) atIndex:4];
    [enc setBytes:&kk length:sizeof(kk) atIndex:5];
    [enc setBytes:&nn length:sizeof(nn) atIndex:6];
  });
}
