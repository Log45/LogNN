#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"

bash "$ROOT_DIR/build_swig_go.sh"

EXT="so"
if [ "$(uname -s)" = "Darwin" ]; then
  EXT="dylib"
fi

NATIVE_DIR="$ROOT_DIR/swig/go/native"
SMOKE_BIN="$ROOT_DIR/swig/go/smoke/smoke"

if [ "$(uname -s)" = "Darwin" ]; then
  DYLD_LIBRARY_PATH="$NATIVE_DIR:${DYLD_LIBRARY_PATH:-}" "$SMOKE_BIN"
else
  LD_LIBRARY_PATH="$NATIVE_DIR:${LD_LIBRARY_PATH:-}" "$SMOKE_BIN"
fi
