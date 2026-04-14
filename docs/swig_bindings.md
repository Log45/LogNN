# SWIG Bindings

SWIG targets (`csharp`, `java`, `go`) are maintained for callable parity with `lognn.cc` pybind API surface.

## Parity target definition

- Parity scope: every callable symbol exposed in `lognn.cc`.
- SWIG parity contract: `docs/swig_parity_contract.md` + machine-readable `docs/swig_parity_contract.json`.
- Deferred item: Python pickle exactness for `save_model/load_model`; SWIG keeps callable names but uses binary backend (`model_io.*`).

## Interface layout

- Shared interface: `swig/lognn_swig.i`
- Shared STL/shared_ptr setup: `swig/common_types.i`
- Language fronts:
  - `swig/lognn_csharp.i`
  - `swig/lognn_java.i`
  - `swig/lognn_go.i`

## Output layout contract

- `swig/csharp/generated`, `swig/csharp/native`, `swig/csharp/Smoke`
- `swig/java/generated`, `swig/java/native`, `swig/java/Smoke`
- `swig/go/generated`, `swig/go/native`, `swig/go/smoke`

## Build + parity checks

- C#: `bash build_swig_csharp.sh` then `bash run_csharp_smoke.sh`
- Java: `bash build_swig_java.sh` then `bash run_java_smoke.sh`
- Go: `bash build_swig_go.sh` then `bash run_go_smoke.sh`
- All languages gate: `bash validate_swig_parity.sh`

All build scripts fail fast with toolchain checks (`swig`, plus language toolchain commands).

## Platform caveats

- Linux runtime path: `LD_LIBRARY_PATH` must include `swig/<lang>/native`.
- macOS runtime path: `DYLD_LIBRARY_PATH` must include `swig/<lang>/native`.
