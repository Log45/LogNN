#!/bin/bash
set -e

# Fix broken conda nvcc environment variables
unset NVCC_PREPEND_FLAGS
unset NVCC_PREPEND_FLAGS_BACKUP
unset NVCC_APPEND_FLAGS

rm -f *.o lognn*.so hw3tensor*.so

# On MacBook hardware, prefer MLX profile first.
if [ "$(uname -s)" = "Darwin" ]; then
  HW_MODEL=$(sysctl -n hw.model 2>/dev/null || echo "")
  if [[ "$HW_MODEL" == MacBook* ]]; then
    echo "Detected MacBook model ($HW_MODEL). Attempting MLX build profile first."
    if bash ./compile_mlx.sh; then
      exit 0
    fi
    echo "MLX build failed on MacBook. Falling back to CPU-only build."
    bash ./compile_cpu.sh
    exit 0
  fi
fi

# If nvcc is unavailable, immediately fall back to CPU-only.
if ! command -v nvcc >/dev/null 2>&1; then
  echo "nvcc not found. Falling back to CPU-only build."
  bash ./compile_cpu.sh
  exit 0
fi

# Auto-detect GPU architecture using detect_gpu.cu
ARCH=""
if nvcc -o /tmp/_detect_gpu detect_gpu.cu 2>/dev/null; then
  SM=$(/tmp/_detect_gpu 2>/dev/null)
  if [ -n "$SM" ]; then
    ARCH="-arch=$SM"
    echo "Detected GPU: $SM"
  fi
fi
rm -f /tmp/_detect_gpu

if [ -z "$ARCH" ]; then
  echo "No CUDA device detected. Falling back to CPU-only build."
  bash ./compile_cpu.sh
  exit 0
fi

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
  echo "Set PYTHON_BIN explicitly, e.g. PYTHON_BIN=python bash compile.sh"
  exit 1
fi

PYTHON_CONFIG_BIN="${PYTHON_BIN}-config"
EXT_SUFFIX=$($PYTHON_BIN -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX') or '.so')")
PYBIND_INCLUDES=$($PYTHON_BIN -m pybind11 --includes)
PY_LDFLAGS=""

if command -v "$PYTHON_CONFIG_BIN" >/dev/null 2>&1; then
  PY_LDFLAGS=$($PYTHON_CONFIG_BIN --embed --ldflags 2>/dev/null || $PYTHON_CONFIG_BIN --ldflags)
fi

if [ "$(uname -s)" = "Darwin" ]; then
  # macOS Python extensions should resolve Python symbols at load time.
  PY_LDFLAGS="-undefined dynamic_lookup"
fi

OMP_CXXFLAGS=""
NVCC_OMP_LINK=""
if [ "${WITH_OPENMP:-}" = "1" ]; then
  OMP_CXXFLAGS="-DWITH_OPENMP -fopenmp"
  NVCC_OMP_LINK="-Xcompiler -fopenmp"
  echo "OpenMP enabled (WITH_OPENMP=1)"
fi

echo "Step 1: compiling CUDA kernels..."
nvcc -O3 -std=c++14 -Xcompiler -fPIC $ARCH -c tensor_kernels.cu -o tensor_kernels.o

echo "Step 2: compiling backend dispatch..."
g++ -O3 -Wall -std=c++14 -fPIC -DWITH_CUDA $OMP_CXXFLAGS -c tensor_kernels.cc -o tensor_kernels_backend.o

echo "Step 3: compiling autograd core..."
g++ -O3 -Wall -std=c++14 -fPIC -DWITH_CUDA $OMP_CXXFLAGS -c autograd.cc -o autograd.o

echo "Step 3b: compiling conv/pool helpers..."
g++ -O3 -Wall -std=c++14 -fPIC -DWITH_CUDA $OMP_CXXFLAGS -c conv_impl.cc -o conv_impl.o

echo "Step 4: compiling pybind11 module..."
g++ -O3 -Wall -std=c++14 -fPIC -DWITH_CUDA $OMP_CXXFLAGS $PYBIND_INCLUDES -c lognn.cc -o lognn.o

echo "Step 5: linking..."
nvcc -shared $NVCC_OMP_LINK -o "lognn${EXT_SUFFIX}" lognn.o autograd.o conv_impl.o tensor_kernels.o tensor_kernels_backend.o $PY_LDFLAGS

echo "Build successful!"
