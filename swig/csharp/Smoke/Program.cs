using LogNN;
using System;
using System.IO;

namespace Smoke;

internal static class Program
{
    private static SizeTVector Dims(params uint[] values)
    {
        var v = new SizeTVector();
        foreach (var x in values) v.Add(x);
        return v;
    }

    private static DoubleVector Data(params double[] values)
    {
        var v = new DoubleVector();
        foreach (var x in values) v.Add(x);
        return v;
    }

    private static double Scalar(Variable x) => x.data().get_data()[0];

    private static void Main()
    {
        // Tensor constructor overload + core ops.
        var sparseIdx = new SizeTVectorVector();
        var i0 = Dims(0, 0);
        var i1 = Dims(1, 1);
        sparseIdx.Add(i0);
        sparseIdx.Add(i1);
        var sparseVal = Data(1.0, 2.0);
        var sparseTensor = new Tensor(Dims(2, 2), sparseIdx, sparseVal, "cpu", 0);
        if (sparseTensor.get_data()[3] != 2.0) throw new Exception("Sparse tensor ctor parity failed");

        var matA = Tensor.from_data(Dims(2, 2), Data(1.0, 2.0, 3.0, 4.0), "cpu", 0);
        var matB = Tensor.from_data(Dims(2, 2), Data(5.0, 6.0, 7.0, 8.0), "cpu", 0);
        var matC = matA.matmul(matB);
        if (matC.get_dims()[0] != 2 || matC.get_dims()[1] != 2) throw new Exception("matmul dims mismatch");

        var reshaped = matC.reshape(Dims(1, 4));
        if (reshaped.get_dims()[1] != 4) throw new Exception("reshape parity failed");

        // Deterministic regression check for backward + optimizer.
        var linear = new Linear(2, 1, "cpu", 0);
        var xTensor = Tensor.from_data(Dims(4, 2), Data(
            -1.0, -1.0,
            -1.0,  1.0,
             1.0, -1.0,
             1.0,  1.0), "cpu", 0);
        var yTensor = Tensor.from_data(Dims(4, 1), Data(0.0, 1.0, 1.0, 2.0), "cpu", 0);
        var x = new Variable(xTensor, false);
        var y = new Variable(yTensor, false);
        var opt = new SGD(linear.parameters(), 0.05);

        var pred0 = linear.forward(x);
        var loss0 = lognn_swig.mse_loss(pred0, y);
        var before = Scalar(loss0);

        opt.zero_grad();
        loss0.backward();
        opt.step();

        var pred1 = linear.forward(x);
        var loss1 = lognn_swig.mse_loss(pred1, y);
        var after = Scalar(loss1);

        var grad0 = linear.parameters()[0].grad().get_data()[0];
        if (double.IsNaN(grad0)) throw new Exception("grad parity failed");

        if (double.IsNaN(before) || double.IsNaN(after) || Math.Abs(before - after) < 1e-12)
        {
            throw new Exception($"C# SWIG smoke failed: before={before}, after={after}");
        }

        // Module constructors / plumbing parity surface.
        var mods = new ModulePtrVector();
        var seq = new Sequential(mods);
        _ = seq.parameters();
        var conv = new Conv2d(1, 2, 3, 3, 1, 1, 1, 1, true, "cpu", 0);
        var maxPool = new MaxPool2d(2, 2, 0);
        var avgPool = new AvgPool2d(2, 2, 0);
        var deconv = new ConvTranspose2d(2, 1, 3, 3, 1, 1, 1, 1, 0, 0, true, "cpu", 0);
        var bn = new BatchNorm2d(2, 0.1, 1e-5, "cpu", 0);
        var tel = new TransformerEncoderLayer(4, 0.1, "cpu", 0);
        var te = new TransformerEncoder(1, 4, 0.1, "cpu", 0);
        var ctel = new CausalTransformerEncoderLayer(4, 0.1, "cpu", 0);
        var cte = new CausalTransformerEncoder(1, 4, 0.1, "cpu", 0);

        // Checkpoint parity alias: save_model/load_model.
        var ckptPath = Path.Combine(Path.GetTempPath(), "lognn_csharp_parity.bin");
        lognn_swig.save_model(te, ckptPath);
        lognn_swig.load_model(te, ckptPath);
        File.Delete(ckptPath);

        // Runtime diagnostics parity entry points.
        _ = lognn_swig.is_mlx_native_enabled();
        _ = lognn_swig.mlx_dispatch_count();
        lognn_swig.reset_mlx_dispatch_count();

        Console.WriteLine($"C# SWIG parity smoke passed. Loss before={before:F6}, after={after:F6}");
    }
}
