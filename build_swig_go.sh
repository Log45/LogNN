#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
GO_ROOT="$ROOT_DIR/swig/go"
GEN_DIR="$GO_ROOT/generated"
NATIVE_DIR="$GO_ROOT/native"
SMOKE_DIR="$GO_ROOT/smoke"

if ! command -v swig >/dev/null 2>&1; then
  echo "swig not found. Install SWIG to build Go bindings."
  exit 1
fi
if ! command -v go >/dev/null 2>&1; then
  echo "go not found. Install Go toolchain."
  exit 1
fi

mkdir -p "$GEN_DIR" "$NATIVE_DIR" "$SMOKE_DIR"
rm -f "$GEN_DIR"/* "$NATIVE_DIR"/liblognn_go.* "$ROOT_DIR"/lognn_go_wrap.o "$SMOKE_DIR"/go.sum "$SMOKE_DIR"/smoke

echo "Step 1: compile native core objects (CPU)..."
g++ -O3 -Wall -std=c++14 -fPIC -c "$ROOT_DIR/tensor_kernels.cc" -o "$ROOT_DIR/tensor_kernels_backend.o"
g++ -O3 -Wall -std=c++14 -fPIC -c "$ROOT_DIR/autograd.cc" -o "$ROOT_DIR/autograd.o"
g++ -O3 -Wall -std=c++14 -fPIC -c "$ROOT_DIR/conv_impl.cc" -o "$ROOT_DIR/conv_impl.o"
g++ -O3 -Wall -std=c++14 -fPIC -c "$ROOT_DIR/model_io.cc" -o "$ROOT_DIR/model_io.o"

echo "Step 2: generate SWIG Go wrappers..."
swig -go -cgo -c++ -intgosize 64 -module lognn_go -I"$ROOT_DIR" \
  -o "$GEN_DIR/lognn_go_wrap.cxx" \
  -outdir "$GEN_DIR" \
  "$ROOT_DIR/swig/lognn_go.i"

echo "Step 3: compile Go wrapper..."
g++ -O3 -Wall -std=c++14 -fPIC -I"$ROOT_DIR" -c "$GEN_DIR/lognn_go_wrap.cxx" -o "$ROOT_DIR/lognn_go_wrap.o"

echo "Step 4: link Go native library..."
EXT="so"
if [ "$(uname -s)" = "Darwin" ]; then
  EXT="dylib"
fi
g++ -shared -o "$NATIVE_DIR/liblognn_go.$EXT" \
  "$ROOT_DIR/lognn_go_wrap.o" "$ROOT_DIR/autograd.o" "$ROOT_DIR/conv_impl.o" \
  "$ROOT_DIR/tensor_kernels_backend.o" "$ROOT_DIR/model_io.o"

if [ ! -f "$GO_ROOT/go.mod" ]; then
  (cd "$GO_ROOT" && go mod init lognnswiggo)
fi

echo "Step 5: build Go smoke app..."
(cd "$GO_ROOT" && go mod tidy)
(cd "$SMOKE_DIR" && go build -o smoke .)

echo "SWIG Go build successful."
