#!/usr/bin/env python3
"""
Train a small CNN on MNIST with LogNN and display predictions with OpenCV.

Build LogNN first (e.g. `bash compile_cpu.sh`), then:

  PYTHONPATH=. python3 mnist_opencv_demo.py --device cpu

Writes `mnist_cnn.pkl` (pickle checkpoint) after training unless `--no-save` is set.

Use `--from_checkpoint path.pkl` to skip training and load weights (architecture must match).

Requires: opencv-python, numpy (see requirements.txt).
"""

from __future__ import annotations

import argparse
import gzip
import random
import struct
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

import cv2
import numpy as np

import lognn

MNIST_URLS = {
    "train_images": "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz",
    "train_labels": "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz",
    "test_images": "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz",
    "test_labels": "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz",
}


def _download_cached(url: str, cache_dir: Path) -> bytes:
    cache_dir.mkdir(parents=True, exist_ok=True)
    dst = cache_dir / Path(url).name
    if not dst.exists():
        urllib.request.urlretrieve(url, dst)
    return dst.read_bytes()


def _parse_idx_images(gz_bytes: bytes, limit: int) -> list[float]:
    raw = gzip.decompress(gz_bytes)
    magic, count, rows, cols = struct.unpack(">IIII", raw[:16])
    if magic != 2051:
        raise RuntimeError(f"Unexpected image magic number: {magic}")
    n = min(count, limit)
    size = rows * cols
    out = [0.0] * (n * size)
    start = 16
    for i in range(n * size):
        out[i] = raw[start + i] / 255.0
    return out


def _parse_idx_labels(gz_bytes: bytes, limit: int) -> list[float]:
    raw = gzip.decompress(gz_bytes)
    magic, count = struct.unpack(">II", raw[:8])
    if magic != 2049:
        raise RuntimeError(f"Unexpected label magic number: {magic}")
    n = min(count, limit)
    return [float(v) for v in raw[8 : 8 + n]]


def load_mnist(train_n: int, test_n: int) -> tuple[list[float], list[float], list[float], list[float]]:
    cache_dir = Path(tempfile.gettempdir()) / "lognn_mnist_cache"
    train_x = _parse_idx_images(_download_cached(MNIST_URLS["train_images"], cache_dir), train_n)
    train_y = _parse_idx_labels(_download_cached(MNIST_URLS["train_labels"], cache_dir), train_n)
    test_x = _parse_idx_images(_download_cached(MNIST_URLS["test_images"], cache_dir), test_n)
    test_y = _parse_idx_labels(_download_cached(MNIST_URLS["test_labels"], cache_dir), test_n)
    return train_x, train_y, test_x, test_y


def _batch_xy(images_flat: list[float], labels: list[float], idxs: list[int]) -> tuple[list[float], list[float]]:
    b = len(idxs)
    x = [0.0] * (b * 28 * 28)
    y = [0.0] * b
    for i, idx in enumerate(idxs):
        src = idx * 28 * 28
        dst = i * 28 * 28
        x[dst : dst + 28 * 28] = images_flat[src : src + 28 * 28]
        y[i] = labels[idx]
    return x, y


def _argmax_10(logits_row: list[float]) -> int:
    return max(range(10), key=lambda c: logits_row[c])


def build_model(device: str, device_index: int) -> lognn.nn.Sequential:
    return lognn.nn.Sequential(
        [
            lognn.nn.Conv2d(1, 8, 3, 3, pad_h=1, pad_w=1, device=device, device_index=device_index),
            lognn.nn.ReLU(),
            lognn.nn.MaxPool2d(2, stride=2, padding=0),
            lognn.nn.Conv2d(8, 16, 3, 3, pad_h=1, pad_w=1, device=device, device_index=device_index),
            lognn.nn.ReLU(),
            lognn.nn.MaxPool2d(2, stride=2, padding=0),
            lognn.nn.Flatten(),
            lognn.nn.Linear(16 * 7 * 7, 128, device, device_index),
            lognn.nn.ReLU(),
            lognn.nn.Linear(128, 10, device, device_index),
        ]
    )


def train(
    model: lognn.nn.Sequential,
    opt: lognn.optim.Adam,
    train_x: list[float],
    train_y: list[float],
    device: str,
    device_index: int,
    batch: int,
    steps: int,
    seed: int,
) -> None:
    rng = random.Random(seed)
    n_train = len(train_y)
    model.train()
    for step in range(steps):
        idxs = [rng.randrange(n_train) for _ in range(batch)]
        bx, by = _batch_xy(train_x, train_y, idxs)
        xv = lognn.Variable(lognn.Tensor.from_data([batch, 1, 28, 28], bx, device, device_index), False)
        yt = lognn.Tensor.from_data([batch], by, device, device_index)
        opt.zero_grad()
        logits = model.forward(xv)
        loss = lognn.Variable.cross_entropy_logits(logits, yt)
        loss.backward()
        opt.step()
        if (step + 1) % max(1, steps // 10) == 0 or step == 0:
            v = loss.data().get_data()[0]
            print(f"  step {step + 1}/{steps}  loss={v:.4f}")


def predict_batch(
    model: lognn.nn.Sequential,
    flat_images: list[float],
    n: int,
    device: str,
    device_index: int,
) -> list[int]:
    model.eval()
    xv = lognn.Variable(lognn.Tensor.from_data([n, 1, 28, 28], flat_images, device, device_index), False)
    logits = model.forward(xv).data().get_data()
    preds = []
    for i in range(n):
        row = logits[i * 10 : (i + 1) * 10]
        preds.append(_argmax_10(row))
    return preds


def render_prediction_grid(
    test_x: list[float],
    test_y: list[float],
    preds: list[int],
    n_show: int,
    scale: int,
    margin: int,
) -> np.ndarray:
    """Build a BGR image grid of digits with pred/true labels."""
    n_show = min(n_show, len(test_y))
    cols = min(8, n_show)
    rows = (n_show + cols - 1) // cols
    cell = 28 * scale
    text_h = 40
    tile_h = cell + text_h
    tile_w = cell + 2 * margin
    canvas_h = rows * tile_h + margin
    canvas_w = cols * tile_w + margin
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    canvas[:] = (40, 40, 40)

    for i in range(n_show):
        r, c = divmod(i, cols)
        y0 = margin + r * tile_h
        x0 = margin + c * tile_w
        flat = test_x[i * 784 : (i + 1) * 784]
        g = np.array(flat, dtype=np.float32).reshape(28, 28)
        g_u8 = (np.clip(g, 0.0, 1.0) * 255.0).astype(np.uint8)
        big = cv2.resize(g_u8, (cell, cell), interpolation=cv2.INTER_NEAREST)
        bgr = cv2.cvtColor(big, cv2.COLOR_GRAY2BGR)
        canvas[y0 : y0 + cell, x0 + margin : x0 + margin + cell] = bgr

        truth = int(test_y[i])
        pr = preds[i]
        ok = pr == truth
        color = (0, 220, 0) if ok else (0, 80, 255)
        label = f"pred={pr}  true={truth}"
        cv2.putText(
            canvas,
            label,
            (x0 + margin, y0 + cell + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
    return canvas


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", default="cpu", help='LogNN device: "cpu", "cuda", or "mlx"')
    p.add_argument("--device-index", type=int, default=0)
    p.add_argument("--train-n", type=int, default=4096, help="Number of training images")
    p.add_argument("--test-n", type=int, default=256, help="Number of test images for accuracy + display pool")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--steps", type=int, default=400, help="Training steps")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--display-n", type=int, default=16, help="How many test digits to show in the window")
    p.add_argument("--scale", type=int, default=4, help="Upscale factor for each 28x28 digit")
    p.add_argument("--no-window", action="store_true", help="Skip cv2.imshow; save grid to mnist_predictions.png")
    p.add_argument(
        "--checkpoint",
        default="mnist_cnn.pkl",
        help="Path to save trained weights (LogNN pickle checkpoint; default: mnist_cnn.pkl)",
    )
    p.add_argument("--no-save", action="store_true", help="Do not write checkpoint after training")
    p.add_argument(
        "--from_checkpoint",
        default=None,
        metavar="PATH",
        help="Load weights from a LogNN pickle checkpoint; skip training (same CNN arch as this script)",
    )
    args = p.parse_args()

    device = args.device.strip().lower()
    if device in ("gpu",):
        device = "cuda"

    train_n = 0 if args.from_checkpoint else args.train_n
    print("Loading MNIST (cached under temp)...")
    try:
        train_x, train_y, test_x, test_y = load_mnist(train_n, args.test_n)
    except (urllib.error.URLError, OSError, RuntimeError) as e:
        raise SystemExit(f"Failed to download or parse MNIST: {e}") from e

    print(f"Building model on device={device!r}...")
    model = build_model(device, args.device_index)

    if args.from_checkpoint:
        ckpt_in = Path(args.from_checkpoint)
        if not ckpt_in.is_file():
            raise SystemExit(f"Checkpoint not found: {ckpt_in.resolve()}")
        lognn.load_model(model, str(ckpt_in))
        print(f"Loaded weights from {ckpt_in.resolve()}")
    else:
        opt = lognn.optim.Adam(model.parameters(), lr=args.lr)
        print(f"Training ({args.steps} steps, batch={args.batch})...")
        train(model, opt, train_x, train_y, device, args.device_index, args.batch, args.steps, args.seed)

        if not args.no_save:
            ckpt = Path(args.checkpoint)
            lognn.save_model(model, str(ckpt))
            print(f"Saved trained model to {ckpt.resolve()}")

    print("Evaluating on test set...")
    preds = predict_batch(model, test_x, len(test_y), device, args.device_index)
    correct = sum(1 for i in range(len(test_y)) if preds[i] == int(test_y[i]))
    acc = correct / float(len(test_y))
    print(f"Test accuracy: {acc * 100:.1f}% ({correct}/{len(test_y)})")

    print("Rendering prediction grid...")
    grid = render_prediction_grid(
        test_x,
        test_y,
        preds,
        n_show=args.display_n,
        scale=args.scale,
        margin=8,
    )

    out_path = Path("mnist_predictions.png")
    cv2.imwrite(str(out_path), grid)
    print(f"Saved {out_path.resolve()}")

    if not args.no_window:
        win = "MNIST — LogNN predictions (green=correct)"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.imshow(win, grid)
        print("Press any key in the image window to exit.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
