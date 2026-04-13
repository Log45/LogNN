import lognn


def main():
    x = lognn.Tensor.from_data([4, 1], [0.0, 1.0, 2.0, 3.0], "cpu", 0)
    y = lognn.Tensor.from_data([4, 1], [1.0, 3.0, 5.0, 7.0], "cpu", 0)

    xv = lognn.Variable(x, False)
    yv = lognn.Variable(y, False)

    model = lognn.Linear(1, 1, "cpu", 0)
    optim = lognn.SGD(model.parameters(), 0.01)

    loss_vals = []
    for _ in range(100):
        optim.zero_grad()
        pred = model.forward(xv)
        loss = lognn.mse_loss(pred, yv)
        loss.backward()
        optim.step()
        loss_vals.append(loss.data().get_data()[0])

    print("initial_loss:", loss_vals[0])
    print("final_loss:", loss_vals[-1])
    if loss_vals[-1] >= loss_vals[0]:
        raise RuntimeError("training did not reduce loss")


if __name__ == "__main__":
    main()
