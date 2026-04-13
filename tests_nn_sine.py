#!/usr/bin/env python3
"""
Train a small MLP to approximate sin(x) on [-pi, pi] using LogNN.

Usage (from repo root, after building lognn):
  python tests_nn_sine.py
  python tests_nn_sine.py --device cpu
  python tests_nn_sine.py --device mlx

Logs test MSE on a fixed grid every 50 steps by default (--log-every N, 0 to disable).
"""

from __future__ import annotations

import argparse
import math
import random
import sys

import lognn


def build_model(hidden: int, device: str, device_index: int) -> lognn.nn.Sequential:
    return lognn.nn.Sequential(
        [
            lognn.nn.Linear(1, hidden, device, device_index),
            lognn.nn.Tanh(),
            lognn.nn.Linear(hidden, hidden, device, device_index),
            lognn.nn.Tanh(),
            lognn.nn.Linear(hidden, 1, device, device_index),
        ]
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Fit sin(x) with a LogNN MLP.")
    p.add_argument("--device", type=str, default="cpu", help="cpu, cuda, or mlx")
    p.add_argument("--device-index", type=int, default=0)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--log-every",
        type=int,
        default=50,
        help="Print test MSE on a fixed grid every N steps (0 = never).",
    )
    args = p.parse_args()

    random.seed(args.seed)
    model = build_model(args.hidden, args.device, args.device_index)
    model.train()
    opt = lognn.optim.Adam(model.parameters(), args.lr)

    # Fixed test grid (held out from training minibatches) for monitoring.
    test_n = 64
    test_xs = [(-math.pi + 2.0 * math.pi * i / (test_n - 1)) for i in range(test_n)]
    test_ys = [math.sin(x) for x in test_xs]
    test_xv = lognn.Variable(
        lognn.Tensor.from_data([test_n, 1], test_xs, args.device, args.device_index),
        False,
    )
    test_yv = lognn.Variable(
        lognn.Tensor.from_data([test_n, 1], test_ys, args.device, args.device_index),
        False,
    )

    losses: list[float] = []
    for step in range(args.steps):
        xs = [random.uniform(-math.pi, math.pi) for _ in range(args.batch)]
        ys = [math.sin(x) for x in xs]
        flat_x = []
        for x in xs:
            flat_x.append(x)
        xv = lognn.Variable(
            lognn.Tensor.from_data([args.batch, 1], flat_x, args.device, args.device_index),
            False,
        )
        yv = lognn.Variable(
            lognn.Tensor.from_data([args.batch, 1], ys, args.device, args.device_index),
            False,
        )
        opt.zero_grad()
        pred = model.forward(xv)
        loss = lognn.mse_loss(pred, yv)
        loss.backward()
        opt.step()
        losses.append(loss.data().get_data()[0])

        if args.log_every > 0 and (step + 1) % args.log_every == 0:
            test_pred = model.forward(test_xv)
            test_loss = lognn.mse_loss(test_pred, test_yv).data().get_data()[0]
            print(f"step {step + 1:5d}  test MSE (grid): {test_loss:.6g}")

    # Grid evaluation (no grad needed for metric)
    grid_n = 50
    gx = [(-math.pi + 2.0 * math.pi * i / (grid_n - 1)) for i in range(grid_n)]
    gv = lognn.Variable(
        lognn.Tensor.from_data([grid_n, 1], gx, args.device, args.device_index), False
    )
    with_pred = model.forward(gv).data().get_data()
    mae = sum(abs(with_pred[i] - math.sin(gx[i])) for i in range(grid_n)) / grid_n

    print("sin MLP — initial batch MSE (first step):", losses[0] if losses else None)
    print("sin MLP — final batch MSE (last step):", losses[-1] if losses else None)
    print("sin MLP — mean abs error vs sin on grid:", mae)

    if not losses or losses[-1] >= losses[0]:
        print("Training did not reduce loss.", file=sys.stderr)
        sys.exit(1)
    if mae > 0.08:
        print(f"MAE too high ({mae}); try more steps or different seed.", file=sys.stderr)
        sys.exit(1)
    print("tests_nn_sine: passed")


if __name__ == "__main__":
    main()
