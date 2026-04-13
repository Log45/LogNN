#!/bin/bash
set -e

# Fix broken conda nvcc environment variables
unset NVCC_PREPEND_FLAGS
unset NVCC_PREPEND_FLAGS_BACKUP
unset NVCC_APPEND_FLAGS

rm -f *.o hw3tensor*.so

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
  echo "GPU detection failed, using nvcc default"
fi

EXT_SUFFIX=$(python3-config --extension-suffix)
PYBIND_INCLUDES=$(python3 -m pybind11 --includes)

echo "Step 1: compiling CUDA kernels..."
nvcc -O3 -std=c++11 -Xcompiler -fPIC $ARCH -c tensor_kernels.cu -o tensor_kernels.o

echo "Step 2: compiling pybind11 module..."
g++ -O3 -Wall -std=c++11 -fPIC $PYBIND_INCLUDES -c hw3tensor.cc -o hw3tensor.o

echo "Step 3: linking..."
nvcc -shared -o "hw3tensor${EXT_SUFFIX}" hw3tensor.o tensor_kernels.o

echo "Build successful!"
