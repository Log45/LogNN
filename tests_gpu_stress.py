#!/usr/bin/env python3
"""
Sustained GPU workload for Activity Monitor verification.

Build the MLX extension first (from repo root):
  bash ./compile_mlx.sh

Run (default ~120 seconds of heavy matmul on device=\"mlx\"):
  python tests_gpu_stress.py
  python tests_gpu_stress.py --seconds 300
  python tests_gpu_stress.py --seconds 60 --dim 512 --inner 40

Watch Activity Monitor -> GPU tab while this runs. You should see sustained
GPU activity from the Metal compute path used by device=\"mlx\".
"""

from __future__ import annotations

import argparse
import sys
import time


def main() -> None:
    p = argparse.ArgumentParser(description="Stress MLX/Metal GPU for Activity Monitor.")
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
            "Could not import lognn. Run from the LogNN repo directory and build first:\n"
            "  bash ./compile_mlx.sh\n"
            "  python tests_gpu_stress.py",
            file=sys.stderr,
        )
        raise SystemExit(1) from e

    if not lognn.is_mlx_native_enabled():
        print(
            "MLX native backend is not enabled in this build.\n"
            "Build with: bash ./compile_mlx.sh\n"
            "Then run this script with the same Python that loads lognn*.so from this directory.",
            file=sys.stderr,
        )
        sys.exit(1)

    d = args.dim
    seed_a = 12345
    seed_b = 67890
    # Fixed operands; chain matmul through a running state so the GPU does real work each step.
    a = lognn.Tensor.randn([d, d], "mlx", 0, seed=seed_a)
    b = lognn.Tensor.randn([d, d], "mlx", 0, seed=seed_b)

    lognn.reset_mlx_dispatch_count()
    t0 = time.monotonic()
    t_report = t0
    chunks = 0
    total_inner = 0

    print(
        f"GPU stress: dim={d} inner_matmuls={args.inner} target_duration={args.seconds}s\n"
        "Open Activity Monitor -> GPU and watch while this runs.\n"
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
                disp = lognn.mlx_dispatch_count()
                print(
                    f"  {elapsed:6.1f}s  chunks={chunks}  matmuls≈{total_inner}  mlx_dispatch_count={disp}"
                )
                t_report = now
    except KeyboardInterrupt:
        print("\nStopped by user.")

    elapsed = time.monotonic() - t0
    disp = lognn.mlx_dispatch_count()
    print(
        f"\nDone in {elapsed:.1f}s  chunks={chunks}  matmuls≈{total_inner}  mlx_dispatch_count={disp}"
    )
    if disp == 0:
        print("Warning: mlx_dispatch_count is still 0; GPU path may not have run.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
