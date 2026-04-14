#!/usr/bin/env python3
"""
Stress / timing harness for Conv2d, MaxPool2d, AvgPool2d, and ConvTranspose2d.

Use this to confirm tensors stay on the requested device and to compare wall time
(CPU reference vs GPU-accelerated builds).

Examples:
  PYTHONPATH=. python3 stress_test_conv_layers.py --device cpu
  PYTHONPATH=. python3 stress_test_conv_layers.py --device cuda
  PYTHONPATH=. python3 stress_test_conv_layers.py --device mlx

CUDA: build with compile.sh (WITH_CUDA). Conv/pool/transpose use device kernels when
get_device_type() == \"cuda\".

MLX: build with compile_mlx.sh. Conv/pool/transpose use Metal kernels in tensor_kernels_mlx.mm
(watch mlx_dispatch_count rise during the timed loop). Tensors report \"mlx\".
"""

from __future__ import annotations

import argparse
import os
import sys
import time


def _try_import_lognn():
    try:
        import lognn  # noqa: F401

        return lognn
    except ImportError as e:
        print("Could not import lognn. Build the extension first (e.g. compile_cpu.sh).", file=sys.stderr)
        raise SystemExit(1) from e


def _device_check(lognn, device: str) -> str:
    """Normalize device string; fail early if this binary cannot use it."""
    d = device.lower().strip()
    if d in ("gpu",):
        d = "cuda"
    # Touch parse path by creating a tiny tensor (throws if e.g. cuda without CUDA build).
    try:
        t = lognn.Tensor.zeros([1], d, 0)
        return t.get_device_type()
    except RuntimeError as e:
        print(f"Device '{device}' is not usable with this lognn build: {e}", file=sys.stderr)
        mod = getattr(lognn, "__file__", None)
        if mod:
            print(f"  lognn loaded from: {mod}", file=sys.stderr)
        print(f"  sys.executable: {sys.executable}", file=sys.stderr)
        if d == "mlx":
            nat = getattr(lognn, "is_mlx_native_enabled", lambda: None)()
            print(
                f"  is_mlx_native_enabled() = {nat!r} (False means Metal init failed or wrong binary).",
                file=sys.stderr,
            )
            print(
                "  Hint: build and run with the same interpreter, e.g. "
                "PYTHON_BIN=$(which python3) bash compile_mlx.sh && PYTHONPATH=. python3 stress_test_conv_layers.py --device mlx",
                file=sys.stderr,
            )
        raise SystemExit(2) from e


def run_stress(
    lognn,
    device: str,
    batch: int,
    channels: int,
    spatial: int,
    warmup: int,
    iters: int,
    quiet: bool,
) -> None:
    conv = lognn.nn.Conv2d(
        channels,
        channels * 2,
        3,
        3,
        stride_h=1,
        stride_w=1,
        pad_h=1,
        pad_w=1,
        bias=True,
        device=device,
        device_index=0,
    )
    mp = lognn.nn.MaxPool2d(2, stride=2, padding=0)
    ap = lognn.nn.AvgPool2d(2, stride=2, padding=0)
    cvt = lognn.nn.ConvTranspose2d(
        channels * 2,
        channels,
        4,
        4,
        stride_h=2,
        stride_w=2,
        pad_h=1,
        pad_w=1,
        output_pad_h=0,
        output_pad_w=0,
        bias=True,
        device=device,
        device_index=0,
    )

    x = lognn.Variable(
        lognn.Tensor.randn([batch, channels, spatial, spatial], device, 0, seed=12345),
        requires_grad=True,
    )

    def one_step():
        h = conv.forward(x)
        h = lognn.Variable.relu(h)
        h = mp.forward(h)
        h = ap.forward(h)
        y = cvt.forward(h)
        loss = lognn.Variable.mean(y)
        loss.backward()

    for _ in range(warmup):
        x.zero_grad()
        conv.zero_grad()
        cvt.zero_grad()
        one_step()

    t0 = time.perf_counter()
    for _ in range(iters):
        x.zero_grad()
        conv.zero_grad()
        cvt.zero_grad()
        one_step()
    elapsed = time.perf_counter() - t0

    # Last y from final iter — use fresh forward for shape/device check (no extra backward).
    h = conv.forward(x)
    h = lognn.Variable.relu(h)
    h = mp.forward(h)
    h = ap.forward(h)
    y = cvt.forward(h)

    ms = elapsed / max(iters, 1) * 1000.0
    if quiet:
        print(f"{x.data().get_device_type()}\t{ms:.3f}\tms/iter\tbatch={batch}\tC={channels}\tH=W={spatial}")
        return

    print("--- stress_test_conv_layers ---")
    print(f"build device arg:     {device!r}")
    print(f"x.device:             {x.data().get_device_type()} idx={x.data().get_device_index()}")
    print(f"y.device (last fwd):  {y.data().get_device_type()} idx={y.data().get_device_index()}")
    print(f"x shape:              {x.data().get_dims()}")
    print(f"y shape:              {y.data().get_dims()}")
    print(f"warmup: {warmup}  iters: {iters}  total_fw+bwd time: {elapsed:.4f}s")
    print(f"per iter: {ms:.2f} ms")
    # Help interpret GPU usage
    if hasattr(lognn, "is_mlx_native_enabled"):
        print(f"mlx_native_enabled:   {lognn.is_mlx_native_enabled()}")
    if hasattr(lognn, "mlx_dispatch_count"):
        print(
            "(Metal elementwise dispatch count is unrelated to conv; "
            f"current count={lognn.mlx_dispatch_count()})"
        )
    print()
    print("Notes:")
    print("  - If x/y report \"cuda\", conv and pooling ran on CUDA tensors in this build.")
    print("  - If \"mlx\", conv/pool/transpose run on Metal when this build includes those kernels")
    print("    (see docs/BACKEND_KERNELS.md). Compare timing to --device cpu.")
    print("  - Compare elapsed time: GPU builds should be much faster than CPU at large spatial.")
    if x.data().get_device_type() == "cuda":
        print("  - On Linux/NVIDIA, run `watch -n0.5 nvidia-smi` in another terminal while this runs;")
        print("    GPU utilization should jump during the timed loop.")


def main() -> None:
    lognn = _try_import_lognn()

    default_dev = os.environ.get("LOGNN_STRESS_DEVICE", "").strip()
    if not default_dev:
        default_dev = "cpu"

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--device",
        default=default_dev,
        help='Tensor device: "cpu", "cuda", or "mlx" (default: env LOGNN_STRESS_DEVICE or cpu)',
    )
    p.add_argument("--batch", type=int, default=4, help="N in NCHW")
    p.add_argument("--channels", type=int, default=32, help="C in NCHW")
    p.add_argument("--spatial", type=int, default=128, help="H and W (square input)")
    p.add_argument("--warmup", type=int, default=2, help="Untimed warmup steps")
    p.add_argument("--iters", type=int, default=10, help="Timed iterations (each = full fwd+bwd)")
    p.add_argument("-q", "--quiet", action="store_true", help="Only print one line (timing)")
    args = p.parse_args()

    resolved = _device_check(lognn, args.device)
    if not args.quiet:
        print(f"Resolved device: {resolved!r}\n")

    run_stress(
        lognn,
        resolved,
        batch=args.batch,
        channels=args.channels,
        spatial=args.spatial,
        warmup=args.warmup,
        iters=args.iters,
        quiet=args.quiet,
    )


if __name__ == "__main__":
    main()
