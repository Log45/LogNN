# Backend kernels for new tensor operations

When you add an op that should run on **CPU**, **CUDA**, and **MLX** (Apple Metal), follow this pattern so all three paths stay correct and testable.

## Layout

| Area | CPU | CUDA | MLX (Metal) |
|------|-----|------|-------------|
| Elementwise, matmul, softmax, ... | `tensor_kernels.cc` | `tensor_kernels.cu` | `tensor_kernels_mlx.mm` (`kMetalSource`) |
| Conv2d, pooling, conv transpose | Reference in `conv_impl.cc` (`*_cpu`, vectors) | `tensor_kernels.cu` (`gpu_*_nchw`) | `tensor_kernels_mlx.mm` (`k_mlx_*`, `mlx_*_nchw`) |
| Device dispatch | `conv_impl.cc` (`Tensor` device type) | `#if defined(WITH_CUDA)` | `#if defined(WITH_MLX)` |

Declarations for CUDA live in `tensor_kernels.h` under `#if defined(WITH_CUDA)`. Declarations for MLX conv/pool live under `#if defined(WITH_MLX)`.

## Checklist for a new heavy op

1. **CPU**: Implement reference math (host `std::vector<double>` or direct `Tensor::from_data` / `get_data` as needed). Used when `device.type == CPU` or when no GPU kernel exists.
2. **CUDA**: Add kernels and a `gpu_*` entry point in `tensor_kernels.cu`; declare in `tensor_kernels.h` inside `WITH_CUDA`.
3. **MLX**: Add Metal kernels (float buffers; `double*` tokens map to shared float `MTLBuffer`s) and `mlx_*` wrappers in `tensor_kernels_mlx.mm`; declare in `tensor_kernels.h` inside `WITH_MLX`.
4. **Dispatch**: Branch in the central place (e.g. `conv_impl.cc` or `tensor_kernels.cc` `backend_*`) on `DeviceType::CUDA` / `DeviceType::MLX` before falling back to CPU.
5. **Autograd**: Ensure backward uses the same device path (tensor `grad` and cached state on device where applicable).
6. **Builds**: `compile_cpu.sh` (CPU only), `compile.sh` (CUDA), `compile_mlx.sh` (MLX + Metal).

## MLX notes

- Buffers are **float** on the GPU; host upload/download converts to/from `double` for the Python API.
- Max-pool argmax indices use **uint32** on MLX (`mlx_alloc_u32`); linear indices must fit in 32 bits for very large tensors.
- `backend_mlx_dispatch_count()` increments once per Metal command submission (including new conv/pool kernels).
