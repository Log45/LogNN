#!/bin/bash
set -e

if [ "$(uname -s)" != "Darwin" ]; then
  echo "compile_mlx.sh is macOS-only."
  exit 1
fi

rm -f *.o lognn*.so

if [ -z "${PYTHON_BIN:-}" ]; then
  for cand in python3 python; do
    if command -v "$cand" >/dev/null 2>&1 \
      && "$cand" -c "import pybind11" >/dev/null 2>&1; then
      PYTHON_BIN="$cand"
      break
    fi
  done
fi

if [ -z "${PYTHON_BIN:-}" ]; then
  echo "Could not find a Python interpreter with pybind11 installed."
  echo "Set PYTHON_BIN explicitly, e.g. PYTHON_BIN=python3 bash compile_mlx.sh"
  exit 1
fi

EXT_SUFFIX=$($PYTHON_BIN -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX') or '.so')")
PYBIND_INCLUDES=$($PYTHON_BIN -m pybind11 --includes)

# Optional include/library paths for future native MLX linkage.
MLX_INCLUDE_FLAGS="${MLX_INCLUDE_FLAGS:-}"
MLX_LINK_FLAGS="${MLX_LINK_FLAGS:-}"

echo "Step 1: compiling backend dispatch with Apple backend flags..."
g++ -O3 -Wall -std=c++17 -fPIC -DWITH_MLX $MLX_INCLUDE_FLAGS -c tensor_kernels.cc -o tensor_kernels_backend.o

echo "Step 2: compiling MLX/Metal backend kernels..."
clang++ -O3 -Wall -std=c++17 -fobjc-arc -fPIC -DWITH_MLX $MLX_INCLUDE_FLAGS -c tensor_kernels_mlx.mm -o tensor_kernels_mlx.o

echo "Step 3: compiling autograd core..."
g++ -O3 -Wall -std=c++17 -fPIC -DWITH_MLX $MLX_INCLUDE_FLAGS -c autograd.cc -o autograd.o

echo "Step 3b: compiling conv/pool helpers..."
g++ -O3 -Wall -std=c++17 -fPIC -DWITH_MLX $MLX_INCLUDE_FLAGS -c conv_impl.cc -o conv_impl.o

echo "Step 4: compiling pybind11 module..."
g++ -O3 -Wall -std=c++17 -fPIC -DWITH_MLX $MLX_INCLUDE_FLAGS $PYBIND_INCLUDES -c lognn.cc -o lognn.o

echo "Step 5: linking..."
clang++ -shared -o "lognn${EXT_SUFFIX}" lognn.o autograd.o conv_impl.o tensor_kernels_backend.o tensor_kernels_mlx.o \
  -undefined dynamic_lookup -framework Foundation -framework Metal $MLX_LINK_FLAGS

echo "MLX-profile build successful (native Metal kernel path enabled)."
echo "Extension: lognn${EXT_SUFFIX} — use the same Python to import it, e.g.:"
echo "  PYTHONPATH=. $PYTHON_BIN -c \"import lognn; print(lognn.__file__, lognn.is_mlx_native_enabled())\""
