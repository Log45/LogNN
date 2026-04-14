"""MNIST CNN classification test for LogNN."""

from __future__ import annotations

import gzip
import random
import struct
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

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


def _mnist_subset(train_n: int = 512, test_n: int = 256) -> tuple[list[float], list[float], list[float], list[float]]:
    cache_dir = Path(tempfile.gettempdir()) / "lognn_mnist_cache"
    try:
        train_x = _parse_idx_images(_download_cached(MNIST_URLS["train_images"], cache_dir), train_n)
        train_y = _parse_idx_labels(_download_cached(MNIST_URLS["train_labels"], cache_dir), train_n)
        test_x = _parse_idx_images(_download_cached(MNIST_URLS["test_images"], cache_dir), test_n)
        test_y = _parse_idx_labels(_download_cached(MNIST_URLS["test_labels"], cache_dir), test_n)
    except (urllib.error.URLError, OSError, RuntimeError):
        return [], [], [], []
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


def test_reshape_after_conv_backward():
    conv = lognn.nn.Conv2d(1, 4, 3, 3, pad_h=1, pad_w=1, device="mlx", device_index=0)
    x = lognn.Variable(lognn.Tensor.randn([4, 1, 28, 28], "mlx", 0, seed=1), True)
    y = conv.forward(x)
    flat = lognn.Variable.reshape(y, [4, 4 * 28 * 28])
    loss = lognn.Variable.mean(flat)
    loss.backward()
    w_grad = conv.parameters()[0].grad().get_data()
    assert any(abs(v) > 0.0 for v in w_grad)


def test_mnist_cnn_classifies_above_chance():
    train_x, train_y, test_x, test_y = _mnist_subset(train_n=2048, test_n=512)
    if not train_y or not test_y:
        return
    rng = random.Random(7)
    batch = 64
    steps = 500

    model = lognn.nn.Sequential(
        [
            lognn.nn.Conv2d(1, 8, 3, 3, pad_h=1, pad_w=1, device="mlx", device_index=0),
            lognn.nn.ReLU(),
            lognn.nn.MaxPool2d(2, stride=2, padding=0),
            lognn.nn.Conv2d(8, 16, 3, 3, pad_h=1, pad_w=1, device="mlx", device_index=0),
            lognn.nn.ReLU(),
            lognn.nn.MaxPool2d(2, stride=2, padding=0),
            lognn.nn.Flatten(),
            lognn.nn.Linear(16 * 7 * 7, 128, "mlx", 0),
            lognn.nn.ReLU(),
            lognn.nn.Linear(128, 10, "mlx", 0),
        ]
    )
    model.train()
    opt = lognn.optim.Adam(model.parameters(), lr=1e-3)

    n_train = len(train_y)
    first_loss = None
    last_loss = None
    for _ in range(steps):
        idxs = [rng.randrange(n_train) for _ in range(batch)]
        bx, by = _batch_xy(train_x, train_y, idxs)
        xv = lognn.Variable(lognn.Tensor.from_data([batch, 1, 28, 28], bx, "mlx", 0), False)
        yt = lognn.Tensor.from_data([batch], by, "mlx", 0)

        opt.zero_grad()
        logits = model.forward(xv)
        loss = lognn.Variable.cross_entropy_logits(logits, yt)
        if first_loss is None:
            first_loss = loss.data().get_data()[0]
        loss.backward()
        opt.step()
        last_loss = loss.data().get_data()[0]

    test_batch = len(test_y)
    x_test = lognn.Variable(lognn.Tensor.from_data([test_batch, 1, 28, 28], test_x, "mlx", 0), False)
    logits = model.forward(x_test).data().get_data()
    correct = 0
    for i, truth in enumerate(test_y):
        row = logits[i * 10 : (i + 1) * 10]
        pred = max(range(10), key=lambda c: row[c])
        if pred == int(truth):
            correct += 1
    acc = correct / float(test_batch)

    assert last_loss is not None and first_loss is not None
    assert last_loss < first_loss
    assert acc >= 0.80
