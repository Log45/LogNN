"""Tests for design-doc plan features: nn modules, softmax, dropout, transformer, AdamW, tensor v1."""

import lognn


def test_sequential():
    seq = lognn.nn.Sequential(
        [
            lognn.nn.Linear(2, 3, "cpu", 0),
            lognn.nn.ReLU(),
            lognn.nn.Linear(3, 1, "cpu", 0),
        ]
    )
    x = lognn.Variable(lognn.Tensor.from_data([4, 2], [0.0, 1.0, 1.0, 0.0, 2.0, -1.0, 0.5, 0.5], "cpu", 0), False)
    y = lognn.Variable(lognn.Tensor.from_data([4, 1], [1.0, 0.0, 2.0, 0.5], "cpu", 0), False)
    opt = lognn.optim.SGD(seq.parameters(), 0.05)
    loss0 = None
    loss1 = None
    for i in range(80):
        opt.zero_grad()
        pred = seq.forward(x)
        loss = lognn.mse_loss(pred, y)
        if i == 0:
            loss0 = loss.data().get_data()[0]
        loss.backward()
        opt.step()
        loss1 = loss.data().get_data()[0]
    assert loss1 < loss0, "Sequential training should reduce loss"


def test_softmax_backward():
    x = lognn.Variable(lognn.Tensor.from_data([2, 3], [0.1, 0.2, 0.3, -0.5, 1.0, 0.0], "cpu", 0), True)
    s = lognn.Variable.softmax_last_dim(x)
    loss = lognn.Variable.mean(s)
    loss.backward()
    g = x.grad().get_data()
    assert len(g) == 6 and all(abs(v) < 1e6 for v in g)


def test_dropout_train_eval():
    d = lognn.nn.Dropout(0.5, seed=123)
    x = lognn.Variable(lognn.Tensor.ones([2, 4], "cpu", 0), False)
    d.train()
    y1 = d.forward(x).data().get_data()
    d.eval()
    y2 = d.forward(x).data().get_data()
    assert any(abs(a - 1.0) > 1e-6 for a in y1) or max(y1) > 1.0 + 1e-6
    assert all(abs(a - 1.0) < 1e-9 for a in y2)


def test_adamw():
    w = lognn.Variable(lognn.Tensor.from_data([1, 1], [1.0], "cpu", 0), True)
    opt = lognn.optim.AdamW([w], lr=0.1, weight_decay=0.1)
    opt.zero_grad()
    loss = lognn.Variable.mean(w)
    loss.backward()
    opt.step()
    assert w.data().get_data()[0] != 1.0


def test_tensor_v1():
    z = lognn.Tensor.zeros([2, 3], "cpu", 0)
    assert all(v == 0.0 for v in z.get_data())
    f = lognn.Tensor.full([2], 2.5, "cpu", 0)
    assert f.get_data() == [2.5, 2.5]
    u = lognn.Tensor.from_data([1, 2, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "cpu", 0)
    sq = u.squeeze()
    assert sq.get_dims() == [2, 3]
    a = lognn.Tensor.from_data([2, 2], [1.0, 2.0, 3.0, 4.0], "cpu", 0)
    b = lognn.Tensor.from_data([1, 2], [10.0, 20.0], "cpu", 0)
    c = a.add(b)
    assert abs(c.get_data()[0] - 11.0) < 1e-9


def test_transformer_tiny():
    T, D = 4, 8
    enc = lognn.nn.TransformerEncoder(1, D, 0.0, "cpu", 0)
    enc.train()
    x = lognn.Variable(lognn.Tensor.randn([T, D], "cpu", 0, seed=7), True)
    target = lognn.Variable(lognn.Tensor.randn([T, D], "cpu", 0, seed=8), False)
    opt = lognn.optim.Adam(enc.parameters(), 1e-2)
    loss0 = None
    loss1 = None
    for i in range(30):
        opt.zero_grad()
        pred = enc.forward(x)
        loss = lognn.mse_loss(pred, target)
        if i == 0:
            loss0 = loss.data().get_data()[0]
        loss.backward()
        opt.step()
        loss1 = loss.data().get_data()[0]
    assert loss1 < loss0, "Transformer encoder should reduce MSE on synthetic data"


def test_mlx_cpu_parity():
    if not lognn.is_mlx_native_enabled():
        return
    lognn.reset_mlx_dispatch_count()
    a_cpu = lognn.Tensor.from_data([2, 3], [0.2, -1.0, 3.0, 0.5, 2.0, -0.7], "cpu", 0)
    b_cpu = lognn.Tensor.from_data([3, 2], [1.0, 2.0, -1.0, 0.25, 0.5, -3.0], "cpu", 0)
    out_cpu = a_cpu.matmul(b_cpu).sigmoid().get_data()

    a_mlx = lognn.Tensor.from_data([2, 3], [0.2, -1.0, 3.0, 0.5, 2.0, -0.7], "mlx", 0)
    b_mlx = lognn.Tensor.from_data([3, 2], [1.0, 2.0, -1.0, 0.25, 0.5, -3.0], "mlx", 0)
    out_mlx = a_mlx.matmul(b_mlx).sigmoid().get_data()

    for c, m in zip(out_cpu, out_mlx):
        assert abs(c - m) < 1e-6
    assert lognn.mlx_dispatch_count() > 0, "MLX dispatch counter should increase for GPU kernels"


def test_mlx_train_smoke_step():
    if not lognn.is_mlx_native_enabled():
        return
    lognn.reset_mlx_dispatch_count()
    model = lognn.nn.Linear(2, 1, "mlx", 0)
    x = lognn.Variable(lognn.Tensor.from_data([3, 2], [0.0, 1.0, 1.0, 0.0, 2.0, 2.0], "mlx", 0), False)
    y = lognn.Variable(lognn.Tensor.from_data([3, 1], [1.0, 1.0, 0.0], "mlx", 0), False)
    opt = lognn.optim.SGD(model.parameters(), 0.05)
    opt.zero_grad()
    pred = model.forward(x)
    loss = lognn.mse_loss(pred, y)
    loss.backward()
    before = loss.data().get_data()[0]
    opt.step()
    after = lognn.mse_loss(model.forward(x), y).data().get_data()[0]
    assert abs(after - before) > 1e-12
    assert lognn.mlx_dispatch_count() > 10, "Training step should dispatch multiple MLX kernels"


def main():
    test_sequential()
    test_softmax_backward()
    test_dropout_train_eval()
    test_adamw()
    test_tensor_v1()
    test_transformer_tiny()
    test_mlx_cpu_parity()
    test_mlx_train_smoke_step()
    print("tests_plan: all passed")


if __name__ == "__main__":
    main()
