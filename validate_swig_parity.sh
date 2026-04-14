#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
STRICT="${STRICT_PARITY:-0}"

STATUS_CSHARP="SKIP"
STATUS_JAVA="SKIP"
STATUS_GO="SKIP"

run_check() {
  local name="$1"
  local cmd="$2"
  echo "==> $name"
  if bash -lc "$cmd"; then
    echo "[$name] PASS"
  else
    echo "[$name] FAIL"
    return 1
  fi
}

if command -v swig >/dev/null 2>&1 && command -v dotnet >/dev/null 2>&1; then
  run_check "C# parity smoke" "bash \"$ROOT_DIR/run_csharp_smoke.sh\""
  STATUS_CSHARP="PASS"
elif [ "$STRICT" = "1" ]; then
  echo "[C# parity smoke] FAIL (missing swig or dotnet)"
  exit 1
else
  echo "[C# parity smoke] SKIP (missing swig or dotnet)"
fi

if command -v swig >/dev/null 2>&1 && command -v javac >/dev/null 2>&1 && command -v jar >/dev/null 2>&1; then
  run_check "Java parity smoke" "bash \"$ROOT_DIR/run_java_smoke.sh\""
  STATUS_JAVA="PASS"
elif [ "$STRICT" = "1" ]; then
  echo "[Java parity smoke] FAIL (missing swig/javac/jar)"
  exit 1
else
  echo "[Java parity smoke] SKIP (missing swig/javac/jar)"
fi

if command -v swig >/dev/null 2>&1 && command -v go >/dev/null 2>&1; then
  run_check "Go parity smoke" "bash \"$ROOT_DIR/run_go_smoke.sh\""
  STATUS_GO="PASS"
elif [ "$STRICT" = "1" ]; then
  echo "[Go parity smoke] FAIL (missing swig or go)"
  exit 1
else
  echo "[Go parity smoke] SKIP (missing swig or go)"
fi

echo
echo "SWIG parity matrix (vs docs/swig_parity_contract.json):"
printf "%-14s | %-11s\n" "Language" "Status"
printf "%-14s-+-%-11s\n" "--------------" "-----------"
printf "%-14s | %-11s\n" "csharp" "$STATUS_CSHARP"
printf "%-14s | %-11s\n" "java" "$STATUS_JAVA"
printf "%-14s | %-11s\n" "go" "$STATUS_GO"
echo
echo "Deferred: pickle binary exactness for save_model/load_model (SWIG uses binary model_io backend)."
