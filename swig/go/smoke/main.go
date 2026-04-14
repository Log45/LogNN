package main

import (
	"fmt"
	"math"

	lognn "lognnswiggo/generated"
)

func dims(vals ...uint) lognn.SizeTVector {
	v := lognn.NewSizeTVector()
	for _, x := range vals {
		v.Add(uint64(x))
	}
	return v
}

func data(vals ...float64) lognn.DoubleVector {
	v := lognn.NewDoubleVector()
	for _, x := range vals {
		v.Add(x)
	}
	return v
}

func scalar(v lognn.Variable) float64 {
	d := v.Data().Get_data()
	return d.Get(0)
}

func main() {
	linear := lognn.NewLinear(2, 1, "cpu", 0)
	xTensor := lognn.TensorFrom_data(dims(4, 2), data(
		-1.0, -1.0,
		-1.0, 1.0,
		1.0, -1.0,
		1.0, 1.0), "cpu", 0)
	yTensor := lognn.TensorFrom_data(dims(4, 1), data(0.0, 1.0, 1.0, 2.0), "cpu", 0)
	x := lognn.NewVariable(xTensor, false)
	y := lognn.NewVariable(yTensor, false)
	opt := lognn.NewSGD(linear.Parameters(), 0.05)

	pred0 := linear.Forward(x)
	loss0 := lognn.Mse_loss(pred0, y)
	before := scalar(loss0)
	opt.Zero_grad()
	loss0.Backward()
	opt.Step()

	pred1 := linear.Forward(x)
	loss1 := lognn.Mse_loss(pred1, y)
	after := scalar(loss1)
	if math.IsNaN(before) || math.IsNaN(after) || math.Abs(before-after) < 1e-12 {
		panic(fmt.Sprintf("Go SWIG smoke failed: before=%f after=%f", before, after))
	}
	fmt.Printf("Go SWIG smoke passed. Loss before=%.6f, after=%.6f\n", before, after)
}
