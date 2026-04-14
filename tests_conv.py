"""Tests for Conv2d, pooling, ConvTranspose2d, BatchNorm2d, CE/log-softmax."""

import lognn


def test_conv2d_1x1_matches_input():
    """1x1 conv with weight 1 and zero bias copies NCHW activations."""
    x = lognn.Variable(
        lognn.Tensor.from_data([1, 1, 2, 3], list(range(1, 7)), "cpu", 0),
        True,
    )
    w = lognn.Variable(lognn.Tensor.from_data([1, 1, 1, 1], [1.0], "cpu", 0), True)
    b = lognn.Variable(lognn.Tensor.zeros([1, 1, 1, 1], "cpu", 0), True)
    y = lognn.Variable.conv2d(x, w, b, 1, 1, 0, 0, True)
    assert y.data().get_dims() == [1, 1, 2, 3]
    assert y.data().get_data() == list(range(1, 7))


def test_conv2d_backward_runs():
    x = lognn.Variable(lognn.Tensor.randn([1, 1, 4, 4], "cpu", 0, seed=1), True)
    w = lognn.Variable(lognn.Tensor.randn([2, 1, 3, 3], "cpu", 0, seed=2), True)
    b = lognn.Variable(lognn.Tensor.zeros([1, 2, 1, 1], "cpu", 0), True)
    y = lognn.Variable.conv2d(x, w, b, 1, 1, 0, 0, True)
    loss = lognn.Variable.mean(y)
    loss.backward()
    assert x.grad().get_data() and w.grad().get_data() and b.grad().get_data()


def test_max_pool2d_and_avg_pool2d():
    # [1,1,4,4] ascending values
    data = [float(i) for i in range(16)]
    x = lognn.Variable(lognn.Tensor.from_data([1, 1, 4, 4], data, "cpu", 0), True)
    m = lognn.nn.MaxPool2d(2, stride=2, padding=0)
    y = m.forward(x)
    assert y.data().get_dims() == [1, 1, 2, 2]
    loss = lognn.Variable.mean(y)
    loss.backward()
    assert x.grad().get_data()

    a = lognn.nn.AvgPool2d(2, stride=2, padding=0)
    ya = a.forward(x)
    assert ya.data().get_dims() == [1, 1, 2, 2]
    lognn.Variable.mean(ya).backward()


def test_conv_transpose2d_forward_shape():
    # 1x1 transpose: Ci=1, Co=1, k=1, stride 1 -> output same spatial as input
    x = lognn.Variable(lognn.Tensor.randn([1, 1, 2, 2], "cpu", 0, seed=3), True)
    w = lognn.Variable(lognn.Tensor.from_data([1, 1, 1, 1], [2.0], "cpu", 0), True)
    y = lognn.Variable.conv_transpose2d(x, w, 1, 1, 0, 0, 0, 0)
    assert y.data().get_dims() == [1, 1, 2, 2]
    loss = lognn.Variable.mean(y)
    loss.backward()
    assert x.grad().get_data() and w.grad().get_data()


def test_cross_entropy_logits():
    logits = lognn.Variable(
        lognn.Tensor.from_data([2, 3], [1.0, 0.0, 0.0, 0.0, 2.0, 0.0], "cpu", 0),
        True,
    )
    target = lognn.Tensor.from_data([2], [0.0, 1.0], "cpu", 0)
    loss = lognn.Variable.cross_entropy_logits(logits, target)
    loss.backward()
    g = logits.grad().get_data()
    assert len(g) == 6 and all(abs(v) < 10.0 for v in g)


def test_log_softmax_last_dim():
    x = lognn.Variable(lognn.Tensor.from_data([1, 3], [1.0, 2.0, 3.0], "cpu", 0), True)
    y = lognn.Variable.log_softmax_last_dim(x)
    loss = lognn.Variable.mean(y)
    loss.backward()
    assert x.grad().get_data()


def test_batch_norm2d_train_eval():
    bn = lognn.nn.BatchNorm2d(2, momentum=0.5, eps=1e-5, device="cpu", device_index=0)
    x = lognn.Variable(lognn.Tensor.randn([2, 2, 2, 2], "cpu", 0, seed=9), True)
    bn.train()
    y = bn.forward(x)
    lognn.Variable.mean(y).backward()
    assert bn.gamma.grad().get_data()
    bn.eval()
    y2 = bn.forward(x)
    assert y2.data().get_dims() == [2, 2, 2, 2]


def test_nn_conv2d_module():
    conv = lognn.nn.Conv2d(1, 2, 3, 3, pad_h=0, pad_w=0, device="cpu", device_index=0)
    x = lognn.Variable(lognn.Tensor.randn([1, 1, 5, 5], "cpu", 0, seed=11), True)
    y = conv.forward(x)
    assert y.data().get_dims()[0] == 1 and y.data().get_dims()[1] == 2
    lognn.Variable.mean(y).backward()
