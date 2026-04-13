#!/bin/bash
set -e

rm -f *.o lognn*.so hw3tensor*.so

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
  echo "Set PYTHON_BIN explicitly, e.g. PYTHON_BIN=python bash compile_cpu.sh"
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
if [ "${WITH_OPENMP:-}" = "1" ]; then
  OMP_CXXFLAGS="-DWITH_OPENMP -fopenmp"
  echo "OpenMP enabled (WITH_OPENMP=1)"
fi

echo "Step 1: compiling backend dispatch (CPU-only)..."
g++ -O3 -Wall -std=c++14 -fPIC $OMP_CXXFLAGS -c tensor_kernels.cc -o tensor_kernels_backend.o

echo "Step 2: compiling autograd core..."
g++ -O3 -Wall -std=c++14 -fPIC $OMP_CXXFLAGS -c autograd.cc -o autograd.o

echo "Step 3: compiling pybind11 module (CPU-only)..."
g++ -O3 -Wall -std=c++14 -fPIC $OMP_CXXFLAGS $PYBIND_INCLUDES -c lognn.cc -o lognn.o

echo "Step 4: linking (CPU-only)..."
g++ -shared $OMP_CXXFLAGS -o "lognn${EXT_SUFFIX}" lognn.o autograd.o tensor_kernels_backend.o $PY_LDFLAGS

echo "CPU-only build successful!"
