# SWIG Parity Contract (C#/Java/Go vs `lognn.cc`)

This is the source-of-truth parity contract for SWIG targets against the pybind callable surface in `lognn.cc`.

Status meanings:

- `implemented`: callable in language binding with equivalent behavior/signature intent.
- `deferred`: intentionally not matched yet, with rationale.

## Deferred items

- `save_model` / `load_model` **pickle byte-level exactness** is deferred for SWIG targets.
  - SWIG bindings expose `save_model` / `load_model` names backed by binary helpers (`save_model_binary`, `load_model_binary`).
  - Rationale: keep non-Python runtimes independent from Python pickle runtime coupling.

## Contract data

Machine-readable contract lives in `docs/swig_parity_contract.json`.
