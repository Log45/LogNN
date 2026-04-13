#!/usr/bin/env python3
"""
Sustained CUDA GPU workload (Linux/Windows with NVIDIA GPU, or CUDA-capable setup).

Build the CUDA extension first (from repo root, on a machine with nvcc + GPU):
  bash ./compile.sh

If nvcc or a GPU is missing, compile.sh falls back to CPU-only; this script will
fail fast when CUDA tensors are not actually available.

Run:
  python tests_cuda_stress.py
  python tests_cuda_stress.py --seconds 300
  python tests_cuda_stress.py --seconds 60 --dim 512 --inner 40

Use nvidia-smi in another terminal to watch GPU utilization while this runs:
  watch -n 0.5 nvidia-smi
"""

from __future__ import annotations

import argparse
import sys
import time


def _cuda_ok(lognn) -> bool:
    """Probe that device=cuda works in this build (WITH_CUDA)."""
    try:
        t = lognn.Tensor.zeros([2, 2], "cuda", 0)
        _ = t.get_data()
        return True
    except RuntimeError:
        return False


def main() -> None:
    p = argparse.ArgumentParser(description="Stress LogNN CUDA GPU with sustained matmul.")
    p.add_argument(
        "--seconds",
        type=float,
        default=120.0,
        help="How long to run the GPU loop (default: 120).",
    )
    p.add_argument(
        "--dim",
        type=int,
        default=384,
        help="Square matrix dimension for matmul (default: 384).",
    )
    p.add_argument(
        "--inner",
        type=int,
        default=25,
        help="Matmuls per outer chunk before checking time (default: 25).",
    )
    p.add_argument(
        "--report-every",
        type=float,
        default=5.0,
        help="Print progress every N seconds (default: 5).",
    )
    p.add_argument(
        "--device-index",
        type=int,
        default=0,
        help="CUDA device index (default: 0).",
    )
    args = p.parse_args()

    if args.seconds <= 0:
        print("--seconds must be positive", file=sys.stderr)
        sys.exit(2)
    if args.dim < 32:
        print("--dim should be at least 32 for meaningful load", file=sys.stderr)
        sys.exit(2)
    if args.inner < 1:
        print("--inner must be >= 1", file=sys.stderr)
        sys.exit(2)

    try:
        import lognn
    except ImportError as e:
        print(
            "Could not import lognn. Build from the LogNN repo directory:\n"
            "  bash ./compile.sh   # CUDA machine with nvcc + GPU\n"
            "  python tests_cuda_stress.py",
            file=sys.stderr,
        )
        raise SystemExit(1) from e

    if not _cuda_ok(lognn):
        print(
            "CUDA backend is not usable in this build or runtime.\n"
            "Rebuild on a machine with CUDA toolkit + GPU:\n"
            "  bash ./compile.sh\n"
            "If compile.sh fell back to CPU-only, fix nvcc/GPU detection first.",
            file=sys.stderr,
        )
        sys.exit(1)

    d = args.dim
    dev = "cuda"
    idx = args.device_index
    seed_a = 12345
    seed_b = 67890

    a = lognn.Tensor.randn([d, d], dev, idx, seed=seed_a)
    b = lognn.Tensor.randn([d, d], dev, idx, seed=seed_b)

    t0 = time.monotonic()
    t_report = t0
    chunks = 0
    total_inner = 0

    print(
        f"CUDA GPU stress: device={dev}:{idx} dim={d} inner_matmuls={args.inner} "
        f"target_duration={args.seconds}s\n"
        "In another terminal:  watch -n 0.5 nvidia-smi\n"
        "Press Ctrl+C to stop early.\n"
    )

    try:
        while time.monotonic() - t0 < args.seconds:
            x = a
            for _ in range(args.inner):
                x = x.matmul(b)
            a = x
            chunks += 1
            total_inner += args.inner

            now = time.monotonic()
            if now - t_report >= args.report_every:
                elapsed = now - t0
                print(
                    f"  {elapsed:6.1f}s  chunks={chunks}  matmuls≈{total_inner}"
                )
                t_report = now
    except KeyboardInterrupt:
        print("\nStopped by user.")

    elapsed = time.monotonic() - t0
    print(
        f"\nDone in {elapsed:.1f}s  chunks={chunks}  matmuls≈{total_inner}"
    )
    if total_inner == 0:
        print("Warning: no matmuls completed.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
