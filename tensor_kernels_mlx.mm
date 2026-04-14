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
#include <metal_atomic>
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

// --- NCHW conv / pool / conv-transpose (float; mirrors CUDA reference in tensor_kernels.cu) ---
kernel void k_mlx_nchw_pad(const device float* x [[buffer(0)]], device float* xp [[buffer(1)]], constant uint& N [[buffer(2)]], constant uint& Ci [[buffer(3)]], constant uint& H [[buffer(4)]], constant uint& W [[buffer(5)]], constant uint& ph [[buffer(6)]], constant uint& pw [[buffer(7)]], constant uint& Hp [[buffer(8)]], constant uint& Wp [[buffer(9)]], uint gid [[thread_position_in_grid]]) {
  uint tot = N * Ci * H * W;
  if (gid >= tot) return;
  uint w0 = gid % W;
  uint t = gid / W;
  uint h0 = t % H;
  t /= H;
  uint c = t % Ci;
  uint n = t / Ci;
  uint dst = ((n * Ci + c) * Hp + (h0 + ph)) * Wp + (w0 + pw);
  xp[dst] = x[gid];
}
kernel void k_mlx_im2col(const device float* xp [[buffer(0)]], device float* col [[buffer(1)]], constant uint& N [[buffer(2)]], constant uint& Ci [[buffer(3)]], constant uint& Hp [[buffer(4)]], constant uint& Wp [[buffer(5)]], constant uint& kH [[buffer(6)]], constant uint& kW [[buffer(7)]], constant uint& sh [[buffer(8)]], constant uint& sw [[buffer(9)]], constant uint& Ho [[buffer(10)]], constant uint& Wo [[buffer(11)]], constant uint& M [[buffer(12)]], constant uint& K [[buffer(13)]], uint tid [[thread_position_in_grid]]) {
  if (tid >= M * K) return;
  uint m = tid / K;
  uint kk = tid % K;
  uint rem = m % (Ho * Wo);
  uint n = m / (Ho * Wo);
  uint oh = rem / Wo;
  uint ow = rem % Wo;
  uint kw_i = kk % kW;
  uint t2 = kk / kW;
  uint kh_i = t2 % kH;
  uint ci = t2 / kH;
  uint ih = oh * sh + kh_i;
  uint iw = ow * sw + kw_i;
  col[tid] = xp[((n * Ci + ci) * Hp + ih) * Wp + iw];
}
kernel void k_mlx_scatter_y_col(const device float* ycol [[buffer(0)]], device float* y [[buffer(1)]], constant uint& N [[buffer(2)]], constant uint& Co [[buffer(3)]], constant uint& Ho [[buffer(4)]], constant uint& Wo [[buffer(5)]], constant uint& M [[buffer(6)]], uint tid [[thread_position_in_grid]]) {
  if (tid >= M * Co) return;
  uint m = tid / Co;
  uint co = tid % Co;
  float v = ycol[tid];
  uint rem = m % (Ho * Wo);
  uint n = m / (Ho * Wo);
  uint oh = rem / Wo;
  uint ow = rem % Wo;
  y[((n * Co + co) * Ho + oh) * Wo + ow] = v;
}
kernel void k_mlx_col2im_atomic(const device float* col [[buffer(0)]], device atomic_float* dxp [[buffer(1)]], constant uint& N [[buffer(2)]], constant uint& Ci [[buffer(3)]], constant uint& Hp [[buffer(4)]], constant uint& Wp [[buffer(5)]], constant uint& kH [[buffer(6)]], constant uint& kW [[buffer(7)]], constant uint& sh [[buffer(8)]], constant uint& sw [[buffer(9)]], constant uint& Ho [[buffer(10)]], constant uint& Wo [[buffer(11)]], constant uint& M [[buffer(12)]], constant uint& K [[buffer(13)]], uint tid [[thread_position_in_grid]]) {
  if (tid >= M * K) return;
  uint m = tid / K;
  uint kk = tid % K;
  uint rem = m % (Ho * Wo);
  uint n = m / (Ho * Wo);
  uint oh = rem / Wo;
  uint ow = rem % Wo;
  uint kw_i = kk % kW;
  uint t2 = kk / kW;
  uint kh_i = t2 % kH;
  uint ci = t2 / kH;
  uint ih = oh * sh + kh_i;
  uint iw = ow * sw + kw_i;
  uint idx = ((n * Ci + ci) * Hp + ih) * Wp + iw;
  float v = col[tid];
  atomic_fetch_add_explicit(dxp + idx, v, memory_order_relaxed);
}
kernel void k_mlx_nchw_unpad(const device float* dxp [[buffer(0)]], device float* dx [[buffer(1)]], constant uint& N [[buffer(2)]], constant uint& Ci [[buffer(3)]], constant uint& H [[buffer(4)]], constant uint& W [[buffer(5)]], constant uint& ph [[buffer(6)]], constant uint& pw [[buffer(7)]], constant uint& Hp [[buffer(8)]], constant uint& Wp [[buffer(9)]], uint id [[thread_position_in_grid]]) {
  uint tot = N * Ci * H * W;
  if (id >= tot) return;
  uint w0 = id % W;
  uint t = id / W;
  uint h0 = t % H;
  t /= H;
  uint c = t % Ci;
  uint n = t / Ci;
  uint src = ((n * Ci + c) * Hp + (h0 + ph)) * Wp + (w0 + pw);
  dx[id] = dxp[src];
}
kernel void k_mlx_dcol_from_dy_w(const device float* dy_nchw [[buffer(0)]], const device float* w [[buffer(1)]], device float* dcol [[buffer(2)]], constant uint& M [[buffer(3)]], constant uint& Co [[buffer(4)]], constant uint& K [[buffer(5)]], constant uint& N [[buffer(6)]], constant uint& Ho [[buffer(7)]], constant uint& Wo [[buffer(8)]], uint tid [[thread_position_in_grid]]) {
  if (tid >= M * K) return;
  uint m = tid / K;
  uint k = tid % K;
  uint rem = m % (Ho * Wo);
  uint n = m / (Ho * Wo);
  uint oh = rem / Wo;
  uint ow = rem % Wo;
  float s = 0.0f;
  for (uint co = 0; co < Co; ++co) {
    float dyv = dy_nchw[((n * Co + co) * Ho + oh) * Wo + ow];
    s += dyv * w[co * K + k];
  }
  dcol[tid] = s;
}
kernel void k_mlx_dw_from_dy_col(const device float* dy_nchw [[buffer(0)]], const device float* col [[buffer(1)]], device float* dw [[buffer(2)]], constant uint& M [[buffer(3)]], constant uint& Co [[buffer(4)]], constant uint& K [[buffer(5)]], constant uint& Ho [[buffer(6)]], constant uint& Wo [[buffer(7)]], uint tid [[thread_position_in_grid]]) {
  if (tid >= Co * K) return;
  uint co = tid / K;
  uint k = tid % K;
  float s = 0.0f;
  for (uint m = 0; m < M; ++m) {
    uint rem = m % (Ho * Wo);
    uint n = m / (Ho * Wo);
    uint oh = rem / Wo;
    uint ow = rem % Wo;
    float dyv = dy_nchw[((n * Co + co) * Ho + oh) * Wo + ow];
    s += dyv * col[m * K + k];
  }
  dw[tid] = s;
}
kernel void k_mlx_maxpool_fwd(const device float* xp [[buffer(0)]], device float* y [[buffer(1)]], device uint* argmax [[buffer(2)]], constant uint& N [[buffer(3)]], constant uint& C [[buffer(4)]], constant uint& Hp [[buffer(5)]], constant uint& Wp [[buffer(6)]], constant uint& kH [[buffer(7)]], constant uint& kW [[buffer(8)]], constant uint& sh [[buffer(9)]], constant uint& sw [[buffer(10)]], constant uint& Ho [[buffer(11)]], constant uint& Wo [[buffer(12)]], uint tid [[thread_position_in_grid]]) {
  uint nout = N * C * Ho * Wo;
  if (tid >= nout) return;
  uint ow = tid % Wo;
  uint t = tid / Wo;
  uint oh = t % Ho;
  t /= Ho;
  uint c = t % C;
  uint n = t / C;
  float best = -1e30f;
  uint best_i = 0;
  bool first = true;
  for (uint kh = 0; kh < kH; ++kh) {
    for (uint kw = 0; kw < kW; ++kw) {
      uint ih = oh * sh + kh;
      uint iw = ow * sw + kw;
      uint li = ((n * C + c) * Hp + ih) * Wp + iw;
      float v = xp[li];
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
kernel void k_mlx_maxpool_bwd(const device float* dy [[buffer(0)]], const device uint* argmax [[buffer(1)]], device atomic_float* dxp [[buffer(2)]], constant uint& N [[buffer(3)]], constant uint& C [[buffer(4)]], constant uint& Hp [[buffer(5)]], constant uint& Wp [[buffer(6)]], constant uint& Ho [[buffer(7)]], constant uint& Wo [[buffer(8)]], uint tid [[thread_position_in_grid]]) {
  uint nout = N * C * Ho * Wo;
  if (tid >= nout) return;
  uint a = argmax[tid];
  atomic_fetch_add_explicit(dxp + a, dy[tid], memory_order_relaxed);
}
kernel void k_mlx_avgpool_fwd(const device float* xp [[buffer(0)]], device float* y [[buffer(1)]], constant uint& N [[buffer(2)]], constant uint& C [[buffer(3)]], constant uint& Hp [[buffer(4)]], constant uint& Wp [[buffer(5)]], constant uint& kH [[buffer(6)]], constant uint& kW [[buffer(7)]], constant uint& sh [[buffer(8)]], constant uint& sw [[buffer(9)]], constant uint& Ho [[buffer(10)]], constant uint& Wo [[buffer(11)]], uint tid [[thread_position_in_grid]]) {
  uint nout = N * C * Ho * Wo;
  if (tid >= nout) return;
  uint ow = tid % Wo;
  uint t = tid / Wo;
  uint oh = t % Ho;
  t /= Ho;
  uint c = t % C;
  uint n = t / C;
  float s = 0.0f;
  for (uint kh = 0; kh < kH; ++kh) {
    for (uint kw = 0; kw < kW; ++kw) {
      uint ih = oh * sh + kh;
      uint iw = ow * sw + kw;
      s += xp[((n * C + c) * Hp + ih) * Wp + iw];
    }
  }
  float sc = 1.0f / float(kH * kW);
  y[tid] = s * sc;
}
kernel void k_mlx_avgpool_bwd(const device float* dy [[buffer(0)]], device atomic_float* dxp [[buffer(1)]], constant uint& N [[buffer(2)]], constant uint& C [[buffer(3)]], constant uint& Hp [[buffer(4)]], constant uint& Wp [[buffer(5)]], constant uint& kH [[buffer(6)]], constant uint& kW [[buffer(7)]], constant uint& sh [[buffer(8)]], constant uint& sw [[buffer(9)]], constant uint& Ho [[buffer(10)]], constant uint& Wo [[buffer(11)]], constant float& scale [[buffer(12)]], uint tid [[thread_position_in_grid]]) {
  uint nout = N * C * Ho * Wo;
  if (tid >= nout) return;
  uint ow = tid % Wo;
  uint t = tid / Wo;
  uint oh = t % Ho;
  t /= Ho;
  uint c = t % C;
  uint n = t / C;
  float g = dy[tid] * scale;
  for (uint kh = 0; kh < kH; ++kh) {
    for (uint kw = 0; kw < kW; ++kw) {
      uint ih = oh * sh + kh;
      uint iw = ow * sw + kw;
      uint ix = ((n * C + c) * Hp + ih) * Wp + iw;
      atomic_fetch_add_explicit(dxp + ix, g, memory_order_relaxed);
    }
  }
}
kernel void k_mlx_conv_tr_fwd(const device float* x [[buffer(0)]], const device float* w [[buffer(1)]], device atomic_float* y [[buffer(2)]], constant uint& N [[buffer(3)]], constant uint& Ci [[buffer(4)]], constant uint& Hi [[buffer(5)]], constant uint& Wi [[buffer(6)]], constant uint& Co [[buffer(7)]], constant uint& kH [[buffer(8)]], constant uint& kW [[buffer(9)]], constant uint& sh [[buffer(10)]], constant uint& sw [[buffer(11)]], constant uint& ph [[buffer(12)]], constant uint& pw [[buffer(13)]], constant uint& Ho [[buffer(14)]], constant uint& Wo [[buffer(15)]], uint tid [[thread_position_in_grid]]) {
  uint total = N * Ci * Hi * Wi;
  if (tid >= total) return;
  uint wi = tid % Wi;
  uint t = tid / Wi;
  uint hi = t % Hi;
  t /= Hi;
  uint ci = t % Ci;
  uint n = t / Ci;
  float xv = x[((n * Ci + ci) * Hi + hi) * Wi + wi];
  for (uint co = 0; co < Co; ++co) {
    for (uint kh_i = 0; kh_i < kH; ++kh_i) {
      for (uint kw_i = 0; kw_i < kW; ++kw_i) {
        int ho = int(hi) * int(sh) + int(kh_i) - int(ph);
        int wo = int(wi) * int(sw) + int(kw_i) - int(pw);
        if (ho < 0 || wo < 0 || ho >= int(Ho) || wo >= int(Wo)) continue;
        uint wi_idx = ((ci * Co + co) * kH + kh_i) * kW + kw_i;
        uint yi = ((n * Co + co) * Ho + uint(ho)) * Wo + uint(wo);
        atomic_fetch_add_explicit(y + yi, xv * w[wi_idx], memory_order_relaxed);
      }
    }
  }
}
kernel void k_mlx_conv_tr_bwd(const device float* dy [[buffer(0)]], const device float* x [[buffer(1)]], const device float* w [[buffer(2)]], device atomic_float* dx [[buffer(3)]], device atomic_float* dw [[buffer(4)]], constant uint& N [[buffer(5)]], constant uint& Ci [[buffer(6)]], constant uint& Hi [[buffer(7)]], constant uint& Wi [[buffer(8)]], constant uint& Co [[buffer(9)]], constant uint& kH [[buffer(10)]], constant uint& kW [[buffer(11)]], constant uint& sh [[buffer(12)]], constant uint& sw [[buffer(13)]], constant uint& ph [[buffer(14)]], constant uint& pw [[buffer(15)]], constant uint& Ho [[buffer(16)]], constant uint& Wo [[buffer(17)]], uint tid [[thread_position_in_grid]]) {
  uint total = N * Ci * Hi * Wi;
  if (tid >= total) return;
  uint wi = tid % Wi;
  uint t = tid / Wi;
  uint hi = t % Hi;
  t /= Hi;
  uint ci = t % Ci;
  uint n = t / Ci;
  uint xi = ((n * Ci + ci) * Hi + hi) * Wi + wi;
  float xv = x[xi];
  for (uint co = 0; co < Co; ++co) {
    for (uint kh_i = 0; kh_i < kH; ++kh_i) {
      for (uint kw_i = 0; kw_i < kW; ++kw_i) {
        int ho = int(hi) * int(sh) + int(kh_i) - int(ph);
        int wo = int(wi) * int(sw) + int(kw_i) - int(pw);
        if (ho < 0 || wo < 0 || ho >= int(Ho) || wo >= int(Wo)) continue;
        uint wi_idx = ((ci * Co + co) * kH + kh_i) * kW + kw_i;
        uint yi = ((n * Co + co) * Ho + uint(ho)) * Wo + uint(wo);
        float g = dy[yi];
        if (g == 0.0f) continue;
        atomic_fetch_add_explicit(dx + xi, g * w[wi_idx], memory_order_relaxed);
        atomic_fetch_add_explicit(dw + wi_idx, g * xv, memory_order_relaxed);
      }
    }
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

  uint32_t* alloc_u32(size_t n) {
    ensure_ready();
    id<MTLBuffer> b = [device_ newBufferWithLength:n * sizeof(uint32_t) options:MTLResourceStorageModeShared];
    if (!b) throw std::runtime_error("Failed to allocate uint32 Metal buffer");
    const uintptr_t k = next_u32_key_.fetch_add(1);
    {
      std::lock_guard<std::mutex> lock(mu_);
      u32_buffers_[k] = BufferRecord{b, n};
    }
    return reinterpret_cast<uint32_t*>(k);
  }

  void free_u32(uint32_t* p) {
    std::lock_guard<std::mutex> lock(mu_);
    u32_buffers_.erase(reinterpret_cast<uintptr_t>(p));
  }

  id<MTLBuffer> get_u32_buf(uint32_t* token, size_t min_n = 0) {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = u32_buffers_.find(reinterpret_cast<uintptr_t>(token));
    if (it == u32_buffers_.end()) throw std::runtime_error("Invalid MLX uint32 buffer token");
    if (min_n > 0 && it->second.n < min_n) throw std::runtime_error("MLX uint32 buffer too small");
    return it->second.buffer;
  }

 private:
  AppleMlxContext() {
    device_ = MTLCreateSystemDefaultDevice();
    if (!device_) {
      init_diagnostic_ =
          "MTLCreateSystemDefaultDevice() returned nil (no GPU, VM without Metal, or wrong Python arch — use "
          "arm64 native Python, not x86_64 under Rosetta).";
      return;
    }
    queue_ = [device_ newCommandQueue];
    if (!queue_) {
      init_diagnostic_ = "newCommandQueue failed";
      return;
    }
    NSError* err = nil;
    NSString* src = [NSString stringWithUTF8String:kMetalSource];
    // Default options use an older MSL; atomic_float and related atomics need a recent language version.
    MTLCompileOptions* compile_opts = [[MTLCompileOptions alloc] init];
    compile_opts.languageVersion = MTLLanguageVersion3_0;
    library_ = [device_ newLibraryWithSource:src options:compile_opts error:&err];
    if (!library_) {
      if (err) {
        const char* desc = err.localizedDescription.UTF8String;
        init_diagnostic_ = desc ? std::string(desc) : std::string("newLibraryWithSource failed");
      } else {
        init_diagnostic_ = "newLibraryWithSource failed (no NSError)";
      }
      return;
    }
    ok_ = true;
  }

  void ensure_ready() {
    if (!ok_) {
      std::string msg = "MLX/Metal backend unavailable";
      if (!init_diagnostic_.empty()) {
        msg += ": ";
        msg += init_diagnostic_;
      }
      throw std::runtime_error(msg);
    }
  }

  static uintptr_t ptr_to_key(double* p) { return reinterpret_cast<uintptr_t>(p); }
  static double* key_to_ptr(uintptr_t k) { return reinterpret_cast<double*>(k); }

  bool ok_ = false;
  std::string init_diagnostic_;
  id<MTLDevice> device_ = nil;
  id<MTLCommandQueue> queue_ = nil;
  id<MTLLibrary> library_ = nil;
  std::unordered_map<std::string, id<MTLComputePipelineState>> pipelines_;
  std::unordered_map<uintptr_t, BufferRecord> buffers_;
  std::unordered_map<uintptr_t, BufferRecord> u32_buffers_;
  std::mutex mu_;
  std::atomic<uintptr_t> next_key_{1};
  std::atomic<uintptr_t> next_u32_key_{1000000};
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

uint32_t* mlx_alloc_u32(size_t n) {
  return AppleMlxContext::instance().alloc_u32(n);
}

void mlx_free_u32(uint32_t* p) {
  AppleMlxContext::instance().free_u32(p);
}

void mlx_conv2d_forward_nchw(const double* x, const double* w, double* y, size_t N, size_t Ci, size_t H, size_t W,
                             size_t Co, size_t kH, size_t kW, size_t sh, size_t sw, size_t ph, size_t pw, size_t Ho,
                             size_t Wo) {
  auto& ctx = AppleMlxContext::instance();
  const size_t Hp = H + 2 * ph;
  const size_t Wp = W + 2 * pw;
  const size_t K = Ci * kH * kW;
  const size_t M = N * Ho * Wo;
  double* xp = mlx_alloc(N * Ci * Hp * Wp);
  mlx_zero(xp, N * Ci * Hp * Wp);
  id<MTLBuffer> xb = ctx.get_buffer(const_cast<double*>(x), N * Ci * H * W);
  id<MTLBuffer> xpb = ctx.get_buffer(xp, N * Ci * Hp * Wp);
  uint32_t N32 = static_cast<uint32_t>(N), Ci32 = static_cast<uint32_t>(Ci), H32 = static_cast<uint32_t>(H),
           W32 = static_cast<uint32_t>(W);
  uint32_t ph32 = static_cast<uint32_t>(ph), pw32 = static_cast<uint32_t>(pw), Hp32 = static_cast<uint32_t>(Hp),
           Wp32 = static_cast<uint32_t>(Wp);
  run_1d("k_mlx_nchw_pad", static_cast<uint32_t>(N * Ci * H * W), ^(id<MTLComputeCommandEncoder> enc) {
    [enc setBuffer:xb offset:0 atIndex:0];
    [enc setBuffer:xpb offset:0 atIndex:1];
    [enc setBytes:&N32 length:sizeof(N32) atIndex:2];
    [enc setBytes:&Ci32 length:sizeof(Ci32) atIndex:3];
    [enc setBytes:&H32 length:sizeof(H32) atIndex:4];
    [enc setBytes:&W32 length:sizeof(W32) atIndex:5];
    [enc setBytes:&ph32 length:sizeof(ph32) atIndex:6];
    [enc setBytes:&pw32 length:sizeof(pw32) atIndex:7];
    [enc setBytes:&Hp32 length:sizeof(Hp32) atIndex:8];
    [enc setBytes:&Wp32 length:sizeof(Wp32) atIndex:9];
  });
  double* col = mlx_alloc(M * K);
  id<MTLBuffer> col_b = ctx.get_buffer(col, M * K);
  uint32_t N2 = static_cast<uint32_t>(N), Ci2 = static_cast<uint32_t>(Ci), Hp2 = static_cast<uint32_t>(Hp),
           Wp2 = static_cast<uint32_t>(Wp);
  uint32_t kH32 = static_cast<uint32_t>(kH), kW32 = static_cast<uint32_t>(kW), sh32 = static_cast<uint32_t>(sh),
           sw32 = static_cast<uint32_t>(sw);
  uint32_t Ho32 = static_cast<uint32_t>(Ho), Wo32 = static_cast<uint32_t>(Wo);
  uint32_t M32 = static_cast<uint32_t>(M), K32 = static_cast<uint32_t>(K);
  run_1d("k_mlx_im2col", static_cast<uint32_t>(M * K), ^(id<MTLComputeCommandEncoder> enc) {
    [enc setBuffer:xpb offset:0 atIndex:0];
    [enc setBuffer:col_b offset:0 atIndex:1];
    [enc setBytes:&N2 length:sizeof(N2) atIndex:2];
    [enc setBytes:&Ci2 length:sizeof(Ci2) atIndex:3];
    [enc setBytes:&Hp2 length:sizeof(Hp2) atIndex:4];
    [enc setBytes:&Wp2 length:sizeof(Wp2) atIndex:5];
    [enc setBytes:&kH32 length:sizeof(kH32) atIndex:6];
    [enc setBytes:&kW32 length:sizeof(kW32) atIndex:7];
    [enc setBytes:&sh32 length:sizeof(sh32) atIndex:8];
    [enc setBytes:&sw32 length:sizeof(sw32) atIndex:9];
    [enc setBytes:&Ho32 length:sizeof(Ho32) atIndex:10];
    [enc setBytes:&Wo32 length:sizeof(Wo32) atIndex:11];
    [enc setBytes:&M32 length:sizeof(M32) atIndex:12];
    [enc setBytes:&K32 length:sizeof(K32) atIndex:13];
  });
  mlx_free(xp);
  double* wt = mlx_alloc(K * Co);
  mlx_transpose_2d(w, wt, Co, K);
  double* ycol = mlx_alloc(M * Co);
  mlx_zero(ycol, M * Co);
  mlx_matmul_2d(col, wt, ycol, M, K, Co);
  mlx_free(col);
  mlx_free(wt);
  mlx_zero(y, N * Co * Ho * Wo);
  id<MTLBuffer> ycol_b = ctx.get_buffer(ycol, M * Co);
  id<MTLBuffer> y_b = ctx.get_buffer(y, N * Co * Ho * Wo);
  uint32_t N3 = static_cast<uint32_t>(N), Co3 = static_cast<uint32_t>(Co), Ho3 = static_cast<uint32_t>(Ho),
           Wo3 = static_cast<uint32_t>(Wo);
  uint32_t M3 = static_cast<uint32_t>(M);
  run_1d("k_mlx_scatter_y_col", static_cast<uint32_t>(M * Co), ^(id<MTLComputeCommandEncoder> enc) {
    [enc setBuffer:ycol_b offset:0 atIndex:0];
    [enc setBuffer:y_b offset:0 atIndex:1];
    [enc setBytes:&N3 length:sizeof(N3) atIndex:2];
    [enc setBytes:&Co3 length:sizeof(Co3) atIndex:3];
    [enc setBytes:&Ho3 length:sizeof(Ho3) atIndex:4];
    [enc setBytes:&Wo3 length:sizeof(Wo3) atIndex:5];
    [enc setBytes:&M3 length:sizeof(M3) atIndex:6];
  });
  mlx_free(ycol);
}

void mlx_conv2d_backward_nchw(const double* dy, const double* x, const double* w, double* dx, double* dw, size_t N,
                              size_t Ci, size_t H, size_t W, size_t Co, size_t kH, size_t kW, size_t sh, size_t sw,
                              size_t ph, size_t pw, size_t Ho, size_t Wo) {
  auto& ctx = AppleMlxContext::instance();
  const size_t Hp = H + 2 * ph;
  const size_t Wp = W + 2 * pw;
  const size_t Kdim = Ci * kH * kW;
  const size_t M = N * Ho * Wo;
  double* xp = mlx_alloc(N * Ci * Hp * Wp);
  mlx_zero(xp, N * Ci * Hp * Wp);
  id<MTLBuffer> xb = ctx.get_buffer(const_cast<double*>(x), N * Ci * H * W);
  id<MTLBuffer> xpb = ctx.get_buffer(xp, N * Ci * Hp * Wp);
  uint32_t N32 = static_cast<uint32_t>(N), Ci32 = static_cast<uint32_t>(Ci), H32 = static_cast<uint32_t>(H),
           W32 = static_cast<uint32_t>(W);
  uint32_t ph32 = static_cast<uint32_t>(ph), pw32 = static_cast<uint32_t>(pw), Hp32 = static_cast<uint32_t>(Hp),
           Wp32 = static_cast<uint32_t>(Wp);
  run_1d("k_mlx_nchw_pad", static_cast<uint32_t>(N * Ci * H * W), ^(id<MTLComputeCommandEncoder> enc) {
    [enc setBuffer:xb offset:0 atIndex:0];
    [enc setBuffer:xpb offset:0 atIndex:1];
    [enc setBytes:&N32 length:sizeof(N32) atIndex:2];
    [enc setBytes:&Ci32 length:sizeof(Ci32) atIndex:3];
    [enc setBytes:&H32 length:sizeof(H32) atIndex:4];
    [enc setBytes:&W32 length:sizeof(W32) atIndex:5];
    [enc setBytes:&ph32 length:sizeof(ph32) atIndex:6];
    [enc setBytes:&pw32 length:sizeof(pw32) atIndex:7];
    [enc setBytes:&Hp32 length:sizeof(Hp32) atIndex:8];
    [enc setBytes:&Wp32 length:sizeof(Wp32) atIndex:9];
  });
  double* col = mlx_alloc(M * Kdim);
  id<MTLBuffer> col_b = ctx.get_buffer(col, M * Kdim);
  uint32_t N2 = static_cast<uint32_t>(N), Ci2 = static_cast<uint32_t>(Ci), Hp2 = static_cast<uint32_t>(Hp),
           Wp2 = static_cast<uint32_t>(Wp);
  uint32_t kH32 = static_cast<uint32_t>(kH), kW32 = static_cast<uint32_t>(kW), sh32 = static_cast<uint32_t>(sh),
           sw32 = static_cast<uint32_t>(sw);
  uint32_t Ho32 = static_cast<uint32_t>(Ho), Wo32 = static_cast<uint32_t>(Wo);
  uint32_t M32 = static_cast<uint32_t>(M), K32 = static_cast<uint32_t>(Kdim);
  run_1d("k_mlx_im2col", static_cast<uint32_t>(M * Kdim), ^(id<MTLComputeCommandEncoder> enc) {
    [enc setBuffer:xpb offset:0 atIndex:0];
    [enc setBuffer:col_b offset:0 atIndex:1];
    [enc setBytes:&N2 length:sizeof(N2) atIndex:2];
    [enc setBytes:&Ci2 length:sizeof(Ci2) atIndex:3];
    [enc setBytes:&Hp2 length:sizeof(Hp2) atIndex:4];
    [enc setBytes:&Wp2 length:sizeof(Wp2) atIndex:5];
    [enc setBytes:&kH32 length:sizeof(kH32) atIndex:6];
    [enc setBytes:&kW32 length:sizeof(kW32) atIndex:7];
    [enc setBytes:&sh32 length:sizeof(sh32) atIndex:8];
    [enc setBytes:&sw32 length:sizeof(sw32) atIndex:9];
    [enc setBytes:&Ho32 length:sizeof(Ho32) atIndex:10];
    [enc setBytes:&Wo32 length:sizeof(Wo32) atIndex:11];
    [enc setBytes:&M32 length:sizeof(M32) atIndex:12];
    [enc setBytes:&K32 length:sizeof(K32) atIndex:13];
  });
  mlx_free(xp);
  double* dcol = mlx_alloc(M * Kdim);
  id<MTLBuffer> dy_b = ctx.get_buffer(const_cast<double*>(dy), N * Co * Ho * Wo);
  id<MTLBuffer> w_b = ctx.get_buffer(const_cast<double*>(w), Co * Kdim);
  id<MTLBuffer> dcol_b = ctx.get_buffer(dcol, M * Kdim);
  uint32_t Co32 = static_cast<uint32_t>(Co);
  uint32_t N3 = static_cast<uint32_t>(N), Ho3 = static_cast<uint32_t>(Ho), Wo3 = static_cast<uint32_t>(Wo);
  run_1d("k_mlx_dcol_from_dy_w", static_cast<uint32_t>(M * Kdim), ^(id<MTLComputeCommandEncoder> enc) {
    [enc setBuffer:dy_b offset:0 atIndex:0];
    [enc setBuffer:w_b offset:0 atIndex:1];
    [enc setBuffer:dcol_b offset:0 atIndex:2];
    [enc setBytes:&M32 length:sizeof(M32) atIndex:3];
    [enc setBytes:&Co32 length:sizeof(Co32) atIndex:4];
    [enc setBytes:&K32 length:sizeof(K32) atIndex:5];
    [enc setBytes:&N3 length:sizeof(N3) atIndex:6];
    [enc setBytes:&Ho3 length:sizeof(Ho3) atIndex:7];
    [enc setBytes:&Wo3 length:sizeof(Wo3) atIndex:8];
  });
  id<MTLBuffer> dw_b = ctx.get_buffer(dw, Co * Kdim);
  run_1d("k_mlx_dw_from_dy_col", static_cast<uint32_t>(Co * Kdim), ^(id<MTLComputeCommandEncoder> enc) {
    [enc setBuffer:dy_b offset:0 atIndex:0];
    [enc setBuffer:col_b offset:0 atIndex:1];
    [enc setBuffer:dw_b offset:0 atIndex:2];
    [enc setBytes:&M32 length:sizeof(M32) atIndex:3];
    [enc setBytes:&Co32 length:sizeof(Co32) atIndex:4];
    [enc setBytes:&K32 length:sizeof(K32) atIndex:5];
    [enc setBytes:&Ho3 length:sizeof(Ho3) atIndex:6];
    [enc setBytes:&Wo3 length:sizeof(Wo3) atIndex:7];
  });
  mlx_free(col);
  double* dxp = mlx_alloc(N * Ci * Hp * Wp);
  mlx_zero(dxp, N * Ci * Hp * Wp);
  id<MTLBuffer> dxp_b = ctx.get_buffer(dxp, N * Ci * Hp * Wp);
  id<MTLBuffer> dcol_b2 = ctx.get_buffer(dcol, M * Kdim);
  run_1d("k_mlx_col2im_atomic", static_cast<uint32_t>(M * Kdim), ^(id<MTLComputeCommandEncoder> enc) {
    [enc setBuffer:dcol_b2 offset:0 atIndex:0];
    [enc setBuffer:dxp_b offset:0 atIndex:1];
    [enc setBytes:&N2 length:sizeof(N2) atIndex:2];
    [enc setBytes:&Ci2 length:sizeof(Ci2) atIndex:3];
    [enc setBytes:&Hp2 length:sizeof(Hp2) atIndex:4];
    [enc setBytes:&Wp2 length:sizeof(Wp2) atIndex:5];
    [enc setBytes:&kH32 length:sizeof(kH32) atIndex:6];
    [enc setBytes:&kW32 length:sizeof(kW32) atIndex:7];
    [enc setBytes:&sh32 length:sizeof(sh32) atIndex:8];
    [enc setBytes:&sw32 length:sizeof(sw32) atIndex:9];
    [enc setBytes:&Ho32 length:sizeof(Ho32) atIndex:10];
    [enc setBytes:&Wo32 length:sizeof(Wo32) atIndex:11];
    [enc setBytes:&M32 length:sizeof(M32) atIndex:12];
    [enc setBytes:&K32 length:sizeof(K32) atIndex:13];
  });
  mlx_free(dcol);
  mlx_zero(dx, N * Ci * H * W);
  id<MTLBuffer> dxp_u = ctx.get_buffer(dxp, N * Ci * Hp * Wp);
  id<MTLBuffer> dx_b = ctx.get_buffer(dx, N * Ci * H * W);
  uint32_t H32b = static_cast<uint32_t>(H), W32b = static_cast<uint32_t>(W);
  run_1d("k_mlx_nchw_unpad", static_cast<uint32_t>(N * Ci * H * W), ^(id<MTLComputeCommandEncoder> enc) {
    [enc setBuffer:dxp_u offset:0 atIndex:0];
    [enc setBuffer:dx_b offset:0 atIndex:1];
    [enc setBytes:&N32 length:sizeof(N32) atIndex:2];
    [enc setBytes:&Ci32 length:sizeof(Ci32) atIndex:3];
    [enc setBytes:&H32b length:sizeof(H32b) atIndex:4];
    [enc setBytes:&W32b length:sizeof(W32b) atIndex:5];
    [enc setBytes:&ph32 length:sizeof(ph32) atIndex:6];
    [enc setBytes:&pw32 length:sizeof(pw32) atIndex:7];
    [enc setBytes:&Hp32 length:sizeof(Hp32) atIndex:8];
    [enc setBytes:&Wp32 length:sizeof(Wp32) atIndex:9];
  });
  mlx_free(dxp);
}

void mlx_maxpool2d_forward_nchw(const double* x, double* y, uint32_t* argmax, size_t N, size_t C, size_t H, size_t W,
                                size_t kH, size_t kW, size_t sh, size_t sw, size_t ph, size_t pw, size_t Ho, size_t Wo,
                                size_t Hp, size_t Wp) {
  auto& ctx = AppleMlxContext::instance();
  double* xp = mlx_alloc(N * C * Hp * Wp);
  mlx_zero(xp, N * C * Hp * Wp);
  id<MTLBuffer> xb = ctx.get_buffer(const_cast<double*>(x), N * C * H * W);
  id<MTLBuffer> xpb = ctx.get_buffer(xp, N * C * Hp * Wp);
  uint32_t N32 = static_cast<uint32_t>(N), C32 = static_cast<uint32_t>(C), H32 = static_cast<uint32_t>(H),
           W32 = static_cast<uint32_t>(W);
  uint32_t ph32 = static_cast<uint32_t>(ph), pw32 = static_cast<uint32_t>(pw), Hp32 = static_cast<uint32_t>(Hp),
           Wp32 = static_cast<uint32_t>(Wp);
  run_1d("k_mlx_nchw_pad", static_cast<uint32_t>(N * C * H * W), ^(id<MTLComputeCommandEncoder> enc) {
    [enc setBuffer:xb offset:0 atIndex:0];
    [enc setBuffer:xpb offset:0 atIndex:1];
    [enc setBytes:&N32 length:sizeof(N32) atIndex:2];
    [enc setBytes:&C32 length:sizeof(C32) atIndex:3];
    [enc setBytes:&H32 length:sizeof(H32) atIndex:4];
    [enc setBytes:&W32 length:sizeof(W32) atIndex:5];
    [enc setBytes:&ph32 length:sizeof(ph32) atIndex:6];
    [enc setBytes:&pw32 length:sizeof(pw32) atIndex:7];
    [enc setBytes:&Hp32 length:sizeof(Hp32) atIndex:8];
    [enc setBytes:&Wp32 length:sizeof(Wp32) atIndex:9];
  });
  size_t nout = N * C * Ho * Wo;
  id<MTLBuffer> yb = ctx.get_buffer(y, nout);
  id<MTLBuffer> amb = ctx.get_u32_buf(argmax, nout);
  uint32_t kH32 = static_cast<uint32_t>(kH), kW32 = static_cast<uint32_t>(kW), sh32 = static_cast<uint32_t>(sh),
           sw32 = static_cast<uint32_t>(sw);
  uint32_t Ho32 = static_cast<uint32_t>(Ho), Wo32 = static_cast<uint32_t>(Wo);
  uint32_t Hp2 = static_cast<uint32_t>(Hp), Wp2 = static_cast<uint32_t>(Wp);
  run_1d("k_mlx_maxpool_fwd", static_cast<uint32_t>(nout), ^(id<MTLComputeCommandEncoder> enc) {
    [enc setBuffer:xpb offset:0 atIndex:0];
    [enc setBuffer:yb offset:0 atIndex:1];
    [enc setBuffer:amb offset:0 atIndex:2];
    [enc setBytes:&N32 length:sizeof(N32) atIndex:3];
    [enc setBytes:&C32 length:sizeof(C32) atIndex:4];
    [enc setBytes:&Hp2 length:sizeof(Hp2) atIndex:5];
    [enc setBytes:&Wp2 length:sizeof(Wp2) atIndex:6];
    [enc setBytes:&kH32 length:sizeof(kH32) atIndex:7];
    [enc setBytes:&kW32 length:sizeof(kW32) atIndex:8];
    [enc setBytes:&sh32 length:sizeof(sh32) atIndex:9];
    [enc setBytes:&sw32 length:sizeof(sw32) atIndex:10];
    [enc setBytes:&Ho32 length:sizeof(Ho32) atIndex:11];
    [enc setBytes:&Wo32 length:sizeof(Wo32) atIndex:12];
  });
  mlx_free(xp);
}

void mlx_maxpool2d_backward_nchw(const double* dy, const uint32_t* argmax, double* dx, size_t N, size_t C, size_t H,
                                 size_t W, size_t ph, size_t pw, size_t Ho, size_t Wo, size_t Hp, size_t Wp) {
  auto& ctx = AppleMlxContext::instance();
  double* dxp = mlx_alloc(N * C * Hp * Wp);
  mlx_zero(dxp, N * C * Hp * Wp);
  size_t nout = N * C * Ho * Wo;
  id<MTLBuffer> dyb = ctx.get_buffer(const_cast<double*>(dy), nout);
  id<MTLBuffer> amb = ctx.get_u32_buf(const_cast<uint32_t*>(argmax), nout);
  id<MTLBuffer> dxpb = ctx.get_buffer(dxp, N * C * Hp * Wp);
  uint32_t N32 = static_cast<uint32_t>(N), C32 = static_cast<uint32_t>(C), Hp32 = static_cast<uint32_t>(Hp),
           Wp32 = static_cast<uint32_t>(Wp);
  uint32_t Ho32 = static_cast<uint32_t>(Ho), Wo32 = static_cast<uint32_t>(Wo);
  run_1d("k_mlx_maxpool_bwd", static_cast<uint32_t>(nout), ^(id<MTLComputeCommandEncoder> enc) {
    [enc setBuffer:dyb offset:0 atIndex:0];
    [enc setBuffer:amb offset:0 atIndex:1];
    [enc setBuffer:dxpb offset:0 atIndex:2];
    [enc setBytes:&N32 length:sizeof(N32) atIndex:3];
    [enc setBytes:&C32 length:sizeof(C32) atIndex:4];
    [enc setBytes:&Hp32 length:sizeof(Hp32) atIndex:5];
    [enc setBytes:&Wp32 length:sizeof(Wp32) atIndex:6];
    [enc setBytes:&Ho32 length:sizeof(Ho32) atIndex:7];
    [enc setBytes:&Wo32 length:sizeof(Wo32) atIndex:8];
  });
  mlx_zero(dx, N * C * H * W);
  id<MTLBuffer> dxp_u = ctx.get_buffer(dxp, N * C * Hp * Wp);
  id<MTLBuffer> dx_b = ctx.get_buffer(dx, N * C * H * W);
  uint32_t H32 = static_cast<uint32_t>(H), W32 = static_cast<uint32_t>(W);
  uint32_t ph32 = static_cast<uint32_t>(ph), pw32 = static_cast<uint32_t>(pw);
  run_1d("k_mlx_nchw_unpad", static_cast<uint32_t>(N * C * H * W), ^(id<MTLComputeCommandEncoder> enc) {
    [enc setBuffer:dxp_u offset:0 atIndex:0];
    [enc setBuffer:dx_b offset:0 atIndex:1];
    [enc setBytes:&N32 length:sizeof(N32) atIndex:2];
    [enc setBytes:&C32 length:sizeof(C32) atIndex:3];
    [enc setBytes:&H32 length:sizeof(H32) atIndex:4];
    [enc setBytes:&W32 length:sizeof(W32) atIndex:5];
    [enc setBytes:&ph32 length:sizeof(ph32) atIndex:6];
    [enc setBytes:&pw32 length:sizeof(pw32) atIndex:7];
    [enc setBytes:&Hp32 length:sizeof(Hp32) atIndex:8];
    [enc setBytes:&Wp32 length:sizeof(Wp32) atIndex:9];
  });
  mlx_free(dxp);
}

void mlx_avgpool2d_forward_nchw(const double* x, double* y, size_t N, size_t C, size_t H, size_t W, size_t kH,
                                size_t kW, size_t sh, size_t sw, size_t ph, size_t pw, size_t Ho, size_t Wo, size_t Hp,
                                size_t Wp) {
  auto& ctx = AppleMlxContext::instance();
  double* xp = mlx_alloc(N * C * Hp * Wp);
  mlx_zero(xp, N * C * Hp * Wp);
  id<MTLBuffer> xb = ctx.get_buffer(const_cast<double*>(x), N * C * H * W);
  id<MTLBuffer> xpb = ctx.get_buffer(xp, N * C * Hp * Wp);
  uint32_t N32 = static_cast<uint32_t>(N), C32 = static_cast<uint32_t>(C), H32 = static_cast<uint32_t>(H),
           W32 = static_cast<uint32_t>(W);
  uint32_t ph32 = static_cast<uint32_t>(ph), pw32 = static_cast<uint32_t>(pw), Hp32 = static_cast<uint32_t>(Hp),
           Wp32 = static_cast<uint32_t>(Wp);
  run_1d("k_mlx_nchw_pad", static_cast<uint32_t>(N * C * H * W), ^(id<MTLComputeCommandEncoder> enc) {
    [enc setBuffer:xb offset:0 atIndex:0];
    [enc setBuffer:xpb offset:0 atIndex:1];
    [enc setBytes:&N32 length:sizeof(N32) atIndex:2];
    [enc setBytes:&C32 length:sizeof(C32) atIndex:3];
    [enc setBytes:&H32 length:sizeof(H32) atIndex:4];
    [enc setBytes:&W32 length:sizeof(W32) atIndex:5];
    [enc setBytes:&ph32 length:sizeof(ph32) atIndex:6];
    [enc setBytes:&pw32 length:sizeof(pw32) atIndex:7];
    [enc setBytes:&Hp32 length:sizeof(Hp32) atIndex:8];
    [enc setBytes:&Wp32 length:sizeof(Wp32) atIndex:9];
  });
  size_t nout = N * C * Ho * Wo;
  id<MTLBuffer> yb = ctx.get_buffer(y, nout);
  uint32_t kH32 = static_cast<uint32_t>(kH), kW32 = static_cast<uint32_t>(kW), sh32 = static_cast<uint32_t>(sh),
           sw32 = static_cast<uint32_t>(sw);
  uint32_t Ho32 = static_cast<uint32_t>(Ho), Wo32 = static_cast<uint32_t>(Wo);
  uint32_t Hp2 = static_cast<uint32_t>(Hp), Wp2 = static_cast<uint32_t>(Wp);
  run_1d("k_mlx_avgpool_fwd", static_cast<uint32_t>(nout), ^(id<MTLComputeCommandEncoder> enc) {
    [enc setBuffer:xpb offset:0 atIndex:0];
    [enc setBuffer:yb offset:0 atIndex:1];
    [enc setBytes:&N32 length:sizeof(N32) atIndex:2];
    [enc setBytes:&C32 length:sizeof(C32) atIndex:3];
    [enc setBytes:&Hp2 length:sizeof(Hp2) atIndex:4];
    [enc setBytes:&Wp2 length:sizeof(Wp2) atIndex:5];
    [enc setBytes:&kH32 length:sizeof(kH32) atIndex:6];
    [enc setBytes:&kW32 length:sizeof(kW32) atIndex:7];
    [enc setBytes:&sh32 length:sizeof(sh32) atIndex:8];
    [enc setBytes:&sw32 length:sizeof(sw32) atIndex:9];
    [enc setBytes:&Ho32 length:sizeof(Ho32) atIndex:10];
    [enc setBytes:&Wo32 length:sizeof(Wo32) atIndex:11];
  });
  mlx_free(xp);
}

void mlx_avgpool2d_backward_nchw(const double* dy, double* dx, size_t N, size_t C, size_t H, size_t W, size_t kH,
                                 size_t kW, size_t sh, size_t sw, size_t ph, size_t pw, size_t Ho, size_t Wo, size_t Hp,
                                 size_t Wp) {
  auto& ctx = AppleMlxContext::instance();
  double* dxp = mlx_alloc(N * C * Hp * Wp);
  mlx_zero(dxp, N * C * Hp * Wp);
  size_t nout = N * C * Ho * Wo;
  id<MTLBuffer> dyb = ctx.get_buffer(const_cast<double*>(dy), nout);
  id<MTLBuffer> dxpb = ctx.get_buffer(dxp, N * C * Hp * Wp);
  uint32_t N32 = static_cast<uint32_t>(N), C32 = static_cast<uint32_t>(C), Hp32 = static_cast<uint32_t>(Hp),
           Wp32 = static_cast<uint32_t>(Wp);
  uint32_t kH32 = static_cast<uint32_t>(kH), kW32 = static_cast<uint32_t>(kW), sh32 = static_cast<uint32_t>(sh),
           sw32 = static_cast<uint32_t>(sw);
  uint32_t Ho32 = static_cast<uint32_t>(Ho), Wo32 = static_cast<uint32_t>(Wo);
  float sc = 1.0f / static_cast<float>(kH * kW);
  run_1d("k_mlx_avgpool_bwd", static_cast<uint32_t>(nout), ^(id<MTLComputeCommandEncoder> enc) {
    [enc setBuffer:dyb offset:0 atIndex:0];
    [enc setBuffer:dxpb offset:0 atIndex:1];
    [enc setBytes:&N32 length:sizeof(N32) atIndex:2];
    [enc setBytes:&C32 length:sizeof(C32) atIndex:3];
    [enc setBytes:&Hp32 length:sizeof(Hp32) atIndex:4];
    [enc setBytes:&Wp32 length:sizeof(Wp32) atIndex:5];
    [enc setBytes:&kH32 length:sizeof(kH32) atIndex:6];
    [enc setBytes:&kW32 length:sizeof(kW32) atIndex:7];
    [enc setBytes:&sh32 length:sizeof(sh32) atIndex:8];
    [enc setBytes:&sw32 length:sizeof(sw32) atIndex:9];
    [enc setBytes:&Ho32 length:sizeof(Ho32) atIndex:10];
    [enc setBytes:&Wo32 length:sizeof(Wo32) atIndex:11];
    [enc setBytes:&sc length:sizeof(sc) atIndex:12];
  });
  mlx_zero(dx, N * C * H * W);
  id<MTLBuffer> dxp_u = ctx.get_buffer(dxp, N * C * Hp * Wp);
  id<MTLBuffer> dx_b = ctx.get_buffer(dx, N * C * H * W);
  uint32_t H32 = static_cast<uint32_t>(H), W32 = static_cast<uint32_t>(W);
  uint32_t ph32 = static_cast<uint32_t>(ph), pw32 = static_cast<uint32_t>(pw);
  run_1d("k_mlx_nchw_unpad", static_cast<uint32_t>(N * C * H * W), ^(id<MTLComputeCommandEncoder> enc) {
    [enc setBuffer:dxp_u offset:0 atIndex:0];
    [enc setBuffer:dx_b offset:0 atIndex:1];
    [enc setBytes:&N32 length:sizeof(N32) atIndex:2];
    [enc setBytes:&C32 length:sizeof(C32) atIndex:3];
    [enc setBytes:&H32 length:sizeof(H32) atIndex:4];
    [enc setBytes:&W32 length:sizeof(W32) atIndex:5];
    [enc setBytes:&ph32 length:sizeof(ph32) atIndex:6];
    [enc setBytes:&pw32 length:sizeof(pw32) atIndex:7];
    [enc setBytes:&Hp32 length:sizeof(Hp32) atIndex:8];
    [enc setBytes:&Wp32 length:sizeof(Wp32) atIndex:9];
  });
  mlx_free(dxp);
}

void mlx_conv_transpose2d_forward_nchw(const double* x, const double* w, double* y, size_t N, size_t Ci, size_t Hi,
                                     size_t Wi, size_t Co, size_t kH, size_t kW, size_t sh, size_t sw, size_t ph,
                                     size_t pw, size_t Ho, size_t Wo) {
  auto& ctx = AppleMlxContext::instance();
  size_t total_out = N * Co * Ho * Wo;
  mlx_zero(y, total_out);
  id<MTLBuffer> xb = ctx.get_buffer(const_cast<double*>(x), N * Ci * Hi * Wi);
  id<MTLBuffer> wb = ctx.get_buffer(const_cast<double*>(w), Ci * Co * kH * kW);
  id<MTLBuffer> yb = ctx.get_buffer(y, total_out);
  uint32_t N32 = static_cast<uint32_t>(N), Ci32 = static_cast<uint32_t>(Ci), Hi32 = static_cast<uint32_t>(Hi),
           Wi32 = static_cast<uint32_t>(Wi);
  uint32_t Co32 = static_cast<uint32_t>(Co), kH32 = static_cast<uint32_t>(kH), kW32 = static_cast<uint32_t>(kW);
  uint32_t sh32 = static_cast<uint32_t>(sh), sw32 = static_cast<uint32_t>(sw), ph32 = static_cast<uint32_t>(ph),
           pw32 = static_cast<uint32_t>(pw);
  uint32_t Ho32 = static_cast<uint32_t>(Ho), Wo32 = static_cast<uint32_t>(Wo);
  uint32_t total_in = static_cast<uint32_t>(N * Ci * Hi * Wi);
  run_1d("k_mlx_conv_tr_fwd", total_in, ^(id<MTLComputeCommandEncoder> enc) {
    [enc setBuffer:xb offset:0 atIndex:0];
    [enc setBuffer:wb offset:0 atIndex:1];
    [enc setBuffer:yb offset:0 atIndex:2];
    [enc setBytes:&N32 length:sizeof(N32) atIndex:3];
    [enc setBytes:&Ci32 length:sizeof(Ci32) atIndex:4];
    [enc setBytes:&Hi32 length:sizeof(Hi32) atIndex:5];
    [enc setBytes:&Wi32 length:sizeof(Wi32) atIndex:6];
    [enc setBytes:&Co32 length:sizeof(Co32) atIndex:7];
    [enc setBytes:&kH32 length:sizeof(kH32) atIndex:8];
    [enc setBytes:&kW32 length:sizeof(kW32) atIndex:9];
    [enc setBytes:&sh32 length:sizeof(sh32) atIndex:10];
    [enc setBytes:&sw32 length:sizeof(sw32) atIndex:11];
    [enc setBytes:&ph32 length:sizeof(ph32) atIndex:12];
    [enc setBytes:&pw32 length:sizeof(pw32) atIndex:13];
    [enc setBytes:&Ho32 length:sizeof(Ho32) atIndex:14];
    [enc setBytes:&Wo32 length:sizeof(Wo32) atIndex:15];
  });
}

void mlx_conv_transpose2d_backward_nchw(const double* dy, const double* x, const double* w, double* dx, double* dw,
                                          size_t N, size_t Ci, size_t Hi, size_t Wi, size_t Co, size_t kH, size_t kW,
                                          size_t sh, size_t sw, size_t ph, size_t pw, size_t Ho, size_t Wo) {
  auto& ctx = AppleMlxContext::instance();
  mlx_zero(dx, N * Ci * Hi * Wi);
  mlx_zero(dw, Ci * Co * kH * kW);
  id<MTLBuffer> dyb = ctx.get_buffer(const_cast<double*>(dy), N * Co * Ho * Wo);
  id<MTLBuffer> xb = ctx.get_buffer(const_cast<double*>(x), N * Ci * Hi * Wi);
  id<MTLBuffer> wb = ctx.get_buffer(const_cast<double*>(w), Ci * Co * kH * kW);
  id<MTLBuffer> dxb = ctx.get_buffer(dx, N * Ci * Hi * Wi);
  id<MTLBuffer> dwb = ctx.get_buffer(dw, Ci * Co * kH * kW);
  uint32_t N32 = static_cast<uint32_t>(N), Ci32 = static_cast<uint32_t>(Ci), Hi32 = static_cast<uint32_t>(Hi),
           Wi32 = static_cast<uint32_t>(Wi);
  uint32_t Co32 = static_cast<uint32_t>(Co), kH32 = static_cast<uint32_t>(kH), kW32 = static_cast<uint32_t>(kW);
  uint32_t sh32 = static_cast<uint32_t>(sh), sw32 = static_cast<uint32_t>(sw), ph32 = static_cast<uint32_t>(ph),
           pw32 = static_cast<uint32_t>(pw);
  uint32_t Ho32 = static_cast<uint32_t>(Ho), Wo32 = static_cast<uint32_t>(Wo);
  uint32_t total_in = static_cast<uint32_t>(N * Ci * Hi * Wi);
  run_1d("k_mlx_conv_tr_bwd", total_in, ^(id<MTLComputeCommandEncoder> enc) {
    [enc setBuffer:dyb offset:0 atIndex:0];
    [enc setBuffer:xb offset:0 atIndex:1];
    [enc setBuffer:wb offset:0 atIndex:2];
    [enc setBuffer:dxb offset:0 atIndex:3];
    [enc setBuffer:dwb offset:0 atIndex:4];
    [enc setBytes:&N32 length:sizeof(N32) atIndex:5];
    [enc setBytes:&Ci32 length:sizeof(Ci32) atIndex:6];
    [enc setBytes:&Hi32 length:sizeof(Hi32) atIndex:7];
    [enc setBytes:&Wi32 length:sizeof(Wi32) atIndex:8];
    [enc setBytes:&Co32 length:sizeof(Co32) atIndex:9];
    [enc setBytes:&kH32 length:sizeof(kH32) atIndex:10];
    [enc setBytes:&kW32 length:sizeof(kW32) atIndex:11];
    [enc setBytes:&sh32 length:sizeof(sh32) atIndex:12];
    [enc setBytes:&sw32 length:sizeof(sw32) atIndex:13];
    [enc setBytes:&ph32 length:sizeof(ph32) atIndex:14];
    [enc setBytes:&pw32 length:sizeof(pw32) atIndex:15];
    [enc setBytes:&Ho32 length:sizeof(Ho32) atIndex:16];
    [enc setBytes:&Wo32 length:sizeof(Wo32) atIndex:17];
  });
}
