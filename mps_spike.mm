#import <Metal/Metal.h>
#include <stdexcept>

// Spike-only helper for the MPS/Metal evaluation todo.
// This is intentionally isolated from the main build and is not linked by
// default compile scripts.
bool lognn_mps_device_available() {
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  return device != nil;
}

void lognn_mps_spike_require_device() {
  if (!lognn_mps_device_available()) {
    throw std::runtime_error("No Metal device available for MPS spike");
  }
}
