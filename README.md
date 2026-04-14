# LogNN
Tensor and Neural Network Library built from scratch for RIT's ML Systems Implementation course

## What This Library Is

LogNN is a lightweight deep learning library with:

- A `Tensor` core that supports CPU, CUDA, and Apple MLX backends (depending on build).
- An autograd engine (`Variable`) for automatic differentiation.
- A small neural network module system (`lognn.nn`) with common layers.
- Optimizers (`SGD`, `Adam`, `AdamW`) and losses for training.
- Python bindings via a compiled `lognn` extension module.

The project is designed for learning and experimentation: concise APIs, readable implementation, and end-to-end model training examples.

## Core Concepts

- **Tensor**: raw n-dimensional data and math ops.
- **Variable**: wraps tensors to track gradients and perform backprop.
- **Module**: layer/model abstraction with `forward`, `parameters`, `train`, and `eval`.
- **Optimizer**: updates trainable parameters using accumulated gradients.

## Build and Setup

### Python (pybind) setup

Prerequisites:

- C++ toolchain (`g++`/`clang++`)
- Python 3 + pip
- `pybind11` installed in the same Python interpreter used for build

Install Python-side requirements:

- `python3 -m pip install -r requirements.txt`
- `python3 -m pip install pybind11`

Build from repo root:

- Dynamic build: `bash compile.sh`
  - On macOS MacBook, it tries MLX first, then falls back to CPU.
  - On Linux/NVIDIA with `nvcc`, it builds CUDA; otherwise CPU fallback.
- CPU build: `bash compile_cpu.sh`
- Apple MLX build (macOS only): `bash compile_mlx.sh`

Run Python tests/examples with the local extension:

- `PYTHONPATH=. python3 tests_plan.py`

If you use a non-default interpreter, set `PYTHON_BIN` during build:

- `PYTHON_BIN=python3.12 bash compile_cpu.sh`

### SWIG setup (C#/Java/Go)

Base prerequisites:

- `swig`
- C++ toolchain (`g++`)

Per-language toolchains:

- C#: `.NET SDK` (`dotnet`)
- Java: JDK (`javac`, `jar`)
- Go: Go toolchain (`go`)

Build + run:

- C#: `bash build_swig_csharp.sh` then `bash run_csharp_smoke.sh`
- Java: `bash build_swig_java.sh` then `bash run_java_smoke.sh`
- Go: `bash build_swig_go.sh` then `bash run_go_smoke.sh`
- Unified parity gate: `bash validate_swig_parity.sh`

## Quick Python Usage

```python
import lognn

x = lognn.Variable(lognn.Tensor.randn([4, 2], "cpu", 0, seed=1), True)
y = lognn.Variable(lognn.Tensor.randn([4, 1], "cpu", 0, seed=2), False)

model = lognn.nn.Sequential([
    lognn.nn.Linear(2, 8, "cpu", 0),
    lognn.nn.ReLU(),
    lognn.nn.Linear(8, 1, "cpu", 0),
])

opt = lognn.optim.Adam(model.parameters(), lr=1e-3)
for _ in range(100):
    opt.zero_grad()
    pred = model.forward(x)
    loss = lognn.mse_loss(pred, y)
    loss.backward()
    opt.step()
```

## CNN and Classification

LogNN includes convolution and pooling support for image models:

- `lognn.nn.Conv2d`
- `lognn.nn.MaxPool2d`
- `lognn.nn.AvgPool2d`
- `lognn.nn.BatchNorm2d`
- `lognn.nn.Flatten`

For a full MNIST example, see `tests_mnist_cnn.py`.

## Saving and Loading Models

Model-only pickle checkpoints are supported:

```python
import lognn

model = lognn.nn.Sequential([
    lognn.nn.Linear(2, 8, "cpu", 0),
    lognn.nn.ReLU(),
    lognn.nn.Linear(8, 1, "cpu", 0),
])

lognn.save_model(model, "model.pkl")
lognn.load_model(model, "model.pkl")
```

`load_model` expects the target model architecture to match parameter count and shapes.

## Useful Example/Test Scripts

- `tests_plan.py`: broad feature smoke tests.
- `tests_conv.py`: conv/pooling and classification-loss tests.
- `tests_mnist_cnn.py`: CNN training on MNIST.
- `tests_checkpoint.py`: model save/load round-trip tests.
- `tests_nn_sine.py`: MLP regression training example.
- `mnist_opencv_demo.py`: train a CNN on MNIST with LogNN and visualize predictions with OpenCV (`python3 -m pip install numpy opencv-python`).

## Additional Docs

- `docs/design_doc.md`
- `docs/STRETCH_GOALS.md`
- `docs/BACKEND_KERNELS.md`
- `docs/swig_bindings.md`

## Acknowledgements

This project uses Kashagra Gupta's HW3 implementation as a baseline. 

This project was implemented largely in collaboration with ChatGPT and the Cursor IDE Agent