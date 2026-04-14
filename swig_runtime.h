#pragma once

#include <cstddef>

bool backend_mlx_native_available();
size_t backend_mlx_dispatch_count();
void backend_mlx_reset_dispatch_count();
