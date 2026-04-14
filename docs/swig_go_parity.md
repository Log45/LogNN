# SWIG Go Parity

Canonical parity contract is tracked in:

- `docs/swig_parity_contract.md`
- `docs/swig_parity_contract.json`

## Go status

- `Tensor`: implemented
- `Variable`: implemented
- `optim` classes (`SGD`, `Adam`, `AdamW`): implemented
- `nn` classes (`Module`, conv/pool, transformer stack): implemented
- Free functions (`relu`, `sigmoid`, `tanh`, `mse_loss`, diagnostics): implemented
- Checkpoints:
  - callable names `save_model` / `load_model`: implemented
  - pickle binary-exactness with Python: deferred (uses binary checkpoint backend)
