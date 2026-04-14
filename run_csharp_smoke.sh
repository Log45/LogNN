#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"

bash "$ROOT_DIR/build_swig_csharp.sh"
dotnet run --project "$ROOT_DIR/swig/csharp/Smoke/Smoke.csproj" -c Release
