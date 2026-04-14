"""Model checkpoint tests using pickle.

Usage:
  PYTHONPATH=. python3 tests_checkpoint.py
"""

from __future__ import annotations

import os
import tempfile

import lognn


def _build_model() -> lognn.nn.Sequential:
    return lognn.nn.Sequential(
        [
            lognn.nn.Linear(2, 8, "cpu", 0),
            lognn.nn.ReLU(),
            lognn.nn.Linear(8, 3, "cpu", 0),
        ]
    )


def _train_a_few_steps(model: lognn.nn.Sequential) -> None:
    x = lognn.Variable(
        lognn.Tensor.from_data(
            [4, 2],
            [
                -1.0,
                -1.0,
                -1.0,
                1.0,
                1.0,
                -1.0,
                1.0,
                1.0,
            ],
            "cpu",
            0,
        ),
        False,
    )
    y = lognn.Variable(
        lognn.Tensor.from_data(
            [4, 3],
            [
                1.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                1.0,
                1.0,
                0.0,
                0.0,
            ],
            "cpu",
            0,
        ),
        False,
    )
    opt = lognn.optim.Adam(model.parameters(), lr=0.01)
    for _ in range(40):
        opt.zero_grad()
        pred = model.forward(x)
        loss = lognn.mse_loss(pred, y)
        loss.backward()
        opt.step()


def test_model_round_trip_pickle() -> None:
    model = _build_model()
    _train_a_few_steps(model)
    model2 = _build_model()

    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        ckpt = f.name
    try:
        lognn.save_model(model, ckpt)
        lognn.load_model(model2, ckpt)
    finally:
        os.remove(ckpt)

    p1 = model.parameters()
    p2 = model2.parameters()
    assert len(p1) == len(p2)
    for a, b in zip(p1, p2):
        assert a.data().get_dims() == b.data().get_dims()
        assert a.data().get_data() == b.data().get_data()


def test_model_output_equivalence_after_load() -> None:
    model = _build_model()
    _train_a_few_steps(model)
    model2 = _build_model()
    x = lognn.Variable(
        lognn.Tensor.from_data([3, 2], [0.5, -1.0, -0.2, 0.3, 1.5, 0.7], "cpu", 0),
        False,
    )

    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        ckpt = f.name
    try:
        before = model.forward(x).data().get_data()
        lognn.save_model(model, ckpt)
        lognn.load_model(model2, ckpt)
        after = model2.forward(x).data().get_data()
    finally:
        os.remove(ckpt)

    assert len(before) == len(after)
    for a, b in zip(before, after):
        assert abs(a - b) < 1e-12


def test_model_load_shape_mismatch_raises() -> None:
    model_a = _build_model()
    model_b = lognn.nn.Sequential(
        [
            lognn.nn.Linear(2, 4, "cpu", 0),
            lognn.nn.ReLU(),
            lognn.nn.Linear(4, 3, "cpu", 0),
        ]
    )
    _train_a_few_steps(model_a)
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        ckpt = f.name
    try:
        lognn.save_model(model_a, ckpt)
        try:
            lognn.load_model(model_b, ckpt)
            assert False, "Expected shape mismatch error"
        except RuntimeError as exc:
            assert "mismatch" in str(exc).lower()
    finally:
        os.remove(ckpt)


if __name__ == "__main__":
    test_model_round_trip_pickle()
    test_model_output_equivalence_after_load()
    test_model_load_shape_mismatch_raises()
    print("checkpoint tests passed")
