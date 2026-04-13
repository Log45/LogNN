#pragma once

// Apple backend rollout decision (plan todo mlx-decisions):
// - Runtime tensor storage remains double in LogNN Tensor.
// - MLX execution uses native Metal kernels (float32 device buffers).
// - Autograd remains LogNN's native Variable graph.
//
// This keeps behavior consistent while we incrementally replace hot ops with
// native MLX/MPS kernels in later iterations.
enum class AppleAutogradStrategy {
  NATIVE_BACKEND_PRIMITIVES = 0,
};

constexpr bool kAppleBackendBridgeUsesTensorDouble = false;
constexpr AppleAutogradStrategy kAppleAutogradStrategy = AppleAutogradStrategy::NATIVE_BACKEND_PRIMITIVES;
