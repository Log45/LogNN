#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
GEN_DIR="$ROOT_DIR/swig/csharp/generated"
NATIVE_DIR="$ROOT_DIR/swig/csharp/native"

if ! command -v swig >/dev/null 2>&1; then
  echo "swig not found. Install SWIG to build C# bindings."
  exit 1
fi
if ! command -v dotnet >/dev/null 2>&1; then
  echo "dotnet not found. Install .NET SDK to build C# bindings."
  exit 1
fi

mkdir -p "$GEN_DIR" "$NATIVE_DIR"
rm -f "$GEN_DIR"/* "$NATIVE_DIR"/liblognn_swig.*

echo "Step 1: compile native core objects (CPU)..."
g++ -O3 -Wall -std=c++14 -fPIC -c "$ROOT_DIR/tensor_kernels.cc" -o "$ROOT_DIR/tensor_kernels_backend.o"
g++ -O3 -Wall -std=c++14 -fPIC -c "$ROOT_DIR/autograd.cc" -o "$ROOT_DIR/autograd.o"
g++ -O3 -Wall -std=c++14 -fPIC -c "$ROOT_DIR/conv_impl.cc" -o "$ROOT_DIR/conv_impl.o"
g++ -O3 -Wall -std=c++14 -fPIC -c "$ROOT_DIR/model_io.cc" -o "$ROOT_DIR/model_io.o"

echo "Step 2: generate SWIG C# wrappers..."
swig -csharp -c++ -namespace LogNN -I"$ROOT_DIR" \
  -o "$GEN_DIR/lognn_swig_wrap.cxx" \
  -outdir "$GEN_DIR" \
  "$ROOT_DIR/swig/lognn_csharp.i"

echo "Step 3: compile wrapper..."
g++ -O3 -Wall -std=c++14 -fPIC -I"$ROOT_DIR" -c "$GEN_DIR/lognn_swig_wrap.cxx" -o "$ROOT_DIR/lognn_swig_wrap.o"

echo "Step 4: link native SWIG library..."
EXT="so"
if [ "$(uname -s)" = "Darwin" ]; then
  EXT="dylib"
fi
g++ -shared -o "$NATIVE_DIR/liblognn_swig.$EXT" \
  "$ROOT_DIR/lognn_swig_wrap.o" "$ROOT_DIR/autograd.o" "$ROOT_DIR/conv_impl.o" \
  "$ROOT_DIR/tensor_kernels_backend.o" "$ROOT_DIR/model_io.o"

echo "Step 5: build C# smoke project..."
dotnet build "$ROOT_DIR/swig/csharp/Smoke/Smoke.csproj" -c Release

echo "SWIG C# build successful."
