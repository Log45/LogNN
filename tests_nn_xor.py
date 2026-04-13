#!/usr/bin/env python3
"""
Train a small MLP to learn XOR using LogNN.

Usage (from repo root, after building lognn):
  python tests_nn_xor.py
  python tests_nn_xor.py --device cpu

Logs XOR MSE every 50 steps by default (--log-every N, 0 to disable).
"""

from __future__ import annotations

import argparse
import sys

import lognn


def build_model(hidden: int, device: str, device_index: int) -> lognn.nn.Sequential:
    # Nonlinearity is required; XOR is not linearly separable.
    return lognn.nn.Sequential(
        [
            lognn.nn.Linear(2, hidden, device, device_index),
            lognn.nn.Tanh(),
            lognn.nn.Linear(hidden, 1, device, device_index),
        ]
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Learn XOR with a LogNN MLP.")
    p.add_argument("--device", type=str, default="cpu", help="cpu, cuda, or mlx")
    p.add_argument("--device-index", type=int, default=0)
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--hidden", type=int, default=16)
    p.add_argument("--lr", type=float, default=5e-2)
    p.add_argument(
        "--log-every",
        type=int,
        default=50,
        help="Print MSE on XOR every N steps (0 = never).",
    )
    args = p.parse_args()

    # XOR truth table: inputs in {0,1}^2, target in {0,1}
    inputs = [
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
    ]
    targets = [0.0, 1.0, 1.0, 0.0]

    flat_in = []
    for row in inputs:
        flat_in.extend(row)
    flat_t = targets[:]

    xv = lognn.Variable(
        lognn.Tensor.from_data([4, 2], flat_in, args.device, args.device_index), False
    )
    yv = lognn.Variable(
        lognn.Tensor.from_data([4, 1], flat_t, args.device, args.device_index), False
    )

    model = build_model(args.hidden, args.device, args.device_index)
    model.train()
    opt = lognn.optim.Adam(model.parameters(), args.lr)

    losses: list[float] = []
    for step in range(args.steps):
        opt.zero_grad()
        pred = model.forward(xv)
        loss = lognn.mse_loss(pred, yv)
        loss.backward()
        opt.step()
        losses.append(loss.data().get_data()[0])

        if args.log_every > 0 and (step + 1) % args.log_every == 0:
            print(f"step {step + 1:5d}  loss (XOR): {losses[-1]:.6g}")

    pred_final = model.forward(xv).data().get_data()
    max_err = max(abs(pred_final[i] - targets[i]) for i in range(4))

    print("XOR MLP — initial MSE:", losses[0] if losses else None)
    print("XOR MLP — final MSE:", losses[-1] if losses else None)
    print("XOR MLP — predictions:", [round(p, 4) for p in pred_final])
    print("XOR MLP — targets:    ", targets)
    print("XOR MLP — max |pred - target|:", max_err)

    if not losses or losses[-1] >= losses[0]:
        print("Training did not reduce loss.", file=sys.stderr)
        sys.exit(1)
    if max_err > 0.15:
        print("XOR not learned well enough; try more steps or larger hidden.", file=sys.stderr)
        sys.exit(1)
    print("tests_nn_xor: passed")


if __name__ == "__main__":
    main()
