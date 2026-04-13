# Stretch goals (from design_doc.md)

These items are **not** implemented in the core library; they are tracked here as follow-on epics.

## Apple Silicon: MLX / MPS

- **MLX (C++/Python)**: Add a backend behind the same `backend_*` seam as CPU/CUDA (e.g. new `DeviceType` or a `MLX` flag), starting with alloc/copy, matmul, and elementwise ops used by training.
- **MPS (Metal)**: Heavier lift (Metal shaders or bridging); consider only after MLX path or as a separate experiment.

## Load an LLM and run inference

- Choose a weight format (e.g. safetensors) and a minimal model spec (layers matching this repo’s ops).
- Add inference-only ops as needed: RoPE, GELU/SiLU, KV-cache, etc.
- Keep tokenization in Python (`tiktoken`, `tokenizers`, etc.) and pass token ids into C++ as `[seq]` or `[1,seq]` tensors.

## Train a code-completion model

- Dataset: file chunks or structured examples (see [arXiv 2504.04030](https://arxiv.org/html/2504.04030v1)).
- Reuse `TransformerEncoder` / optimizers / device backends; training loop can stay in Python driving `lognn`.
