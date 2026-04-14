#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"

bash "$ROOT_DIR/build_swig_java.sh"

EXT="so"
LIB_PATH_VAR="LD_LIBRARY_PATH"
if [ "$(uname -s)" = "Darwin" ]; then
  EXT="dylib"
  LIB_PATH_VAR="DYLD_LIBRARY_PATH"
fi

NATIVE_DIR="$ROOT_DIR/swig/java/native"
BUILD_DIR="$ROOT_DIR/swig/java/build"

if [ "$LIB_PATH_VAR" = "DYLD_LIBRARY_PATH" ]; then
  DYLD_LIBRARY_PATH="$NATIVE_DIR:${DYLD_LIBRARY_PATH:-}" \
    java -Djava.library.path="$NATIVE_DIR" -cp "$BUILD_DIR" Main
else
  LD_LIBRARY_PATH="$NATIVE_DIR:${LD_LIBRARY_PATH:-}" \
    java -Djava.library.path="$NATIVE_DIR" -cp "$BUILD_DIR" Main
fi
