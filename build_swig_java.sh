#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
GEN_DIR="$ROOT_DIR/swig/java/generated"
NATIVE_DIR="$ROOT_DIR/swig/java/native"
BUILD_DIR="$ROOT_DIR/swig/java/build"
SMOKE_SRC="$ROOT_DIR/swig/java/Smoke/Main.java"

if ! command -v swig >/dev/null 2>&1; then
  echo "swig not found. Install SWIG to build Java bindings."
  exit 1
fi
if ! command -v javac >/dev/null 2>&1; then
  echo "javac not found. Install JDK to build Java bindings."
  exit 1
fi
if ! command -v jar >/dev/null 2>&1; then
  echo "jar not found. Install JDK tooling."
  exit 1
fi

JAVA_HOME_DETECTED="${JAVA_HOME:-}"
if [ -z "$JAVA_HOME_DETECTED" ]; then
  if [ "$(uname -s)" = "Darwin" ] && command -v /usr/libexec/java_home >/dev/null 2>&1; then
    JAVA_HOME_DETECTED="$(/usr/libexec/java_home)"
  else
    JAVAC_BIN="$(command -v javac)"
    JAVA_HOME_DETECTED="$(cd "$(dirname "$JAVAC_BIN")/.." && pwd)"
  fi
fi
JNI_INCLUDE_FLAGS="-I$JAVA_HOME_DETECTED/include"
if [ "$(uname -s)" = "Darwin" ]; then
  JNI_INCLUDE_FLAGS="$JNI_INCLUDE_FLAGS -I$JAVA_HOME_DETECTED/include/darwin"
elif [ "$(uname -s)" = "Linux" ]; then
  JNI_INCLUDE_FLAGS="$JNI_INCLUDE_FLAGS -I$JAVA_HOME_DETECTED/include/linux"
fi

mkdir -p "$GEN_DIR" "$NATIVE_DIR" "$BUILD_DIR"
rm -f "$GEN_DIR"/* "$NATIVE_DIR"/liblognn_java.* "$BUILD_DIR"/*.class "$ROOT_DIR"/lognn_java_wrap.o

echo "Step 1: compile native core objects (CPU)..."
g++ -O3 -Wall -std=c++14 -fPIC -c "$ROOT_DIR/tensor_kernels.cc" -o "$ROOT_DIR/tensor_kernels_backend.o"
g++ -O3 -Wall -std=c++14 -fPIC -c "$ROOT_DIR/autograd.cc" -o "$ROOT_DIR/autograd.o"
g++ -O3 -Wall -std=c++14 -fPIC -c "$ROOT_DIR/conv_impl.cc" -o "$ROOT_DIR/conv_impl.o"
g++ -O3 -Wall -std=c++14 -fPIC -c "$ROOT_DIR/model_io.cc" -o "$ROOT_DIR/model_io.o"

echo "Step 2: generate SWIG Java wrappers..."
swig -java -c++ -I"$ROOT_DIR" \
  -o "$GEN_DIR/lognn_java_wrap.cxx" \
  -outdir "$GEN_DIR" \
  "$ROOT_DIR/swig/lognn_java.i"

echo "Step 3: compile JNI wrapper..."
g++ -O3 -Wall -std=c++14 -fPIC -I"$ROOT_DIR" $JNI_INCLUDE_FLAGS -c "$GEN_DIR/lognn_java_wrap.cxx" -o "$ROOT_DIR/lognn_java_wrap.o"

echo "Step 4: link Java native library..."
EXT="so"
if [ "$(uname -s)" = "Darwin" ]; then
  EXT="dylib"
fi
g++ -shared -o "$NATIVE_DIR/liblognn_java.$EXT" \
  "$ROOT_DIR/lognn_java_wrap.o" "$ROOT_DIR/autograd.o" "$ROOT_DIR/conv_impl.o" \
  "$ROOT_DIR/tensor_kernels_backend.o" "$ROOT_DIR/model_io.o"

echo "Step 5: compile Java classes and smoke app..."
javac -d "$BUILD_DIR" "$GEN_DIR"/*.java "$SMOKE_SRC"
jar cf "$ROOT_DIR/swig/java/lognn_java.jar" -C "$BUILD_DIR" .

echo "SWIG Java build successful."
