public class Main {
  static {
    System.loadLibrary("lognn_java");
  }

  private static SizeTVector dims(int... vals) {
    SizeTVector v = new SizeTVector();
    for (int x : vals) v.add((long)x);
    return v;
  }

  private static DoubleVector data(double... vals) {
    DoubleVector v = new DoubleVector();
    for (double x : vals) v.add(x);
    return v;
  }

  private static double scalar(Variable x) {
    DoubleVector dv = new DoubleVector(x.data().get_data());
    return dv.get(0);
  }

  public static void main(String[] args) {
    // Tensor constructor overload + core ops parity.
    SizeTVectorVector sparseIdx = new SizeTVectorVector();
    sparseIdx.add(dims(0, 0));
    sparseIdx.add(dims(1, 1));
    Tensor sparseTensor = new Tensor(dims(2, 2), sparseIdx, data(1.0, 2.0), "cpu", 0);
    if (new DoubleVector(sparseTensor.get_data()).get(3) != 2.0) {
      throw new RuntimeException("Sparse tensor ctor parity failed");
    }
    Tensor matA = Tensor.from_data(dims(2, 2), data(1.0, 2.0, 3.0, 4.0), "cpu", 0);
    Tensor matB = Tensor.from_data(dims(2, 2), data(5.0, 6.0, 7.0, 8.0), "cpu", 0);
    Tensor matC = matA.matmul(matB);
    if (new SizeTVector(matC.get_dims()).get(0) != 2) throw new RuntimeException("matmul dims mismatch");
    Tensor reshaped = matC.reshape(dims(1, 4));
    if (new SizeTVector(reshaped.get_dims()).get(1) != 4) throw new RuntimeException("reshape parity failed");

    Linear linear = new Linear(2, 1, "cpu", 0);
    Tensor xTensor = Tensor.from_data(dims(4, 2), data(
        -1.0, -1.0,
        -1.0, 1.0,
         1.0, -1.0,
         1.0, 1.0), "cpu", 0);
    Tensor yTensor = Tensor.from_data(dims(4, 1), data(0.0, 1.0, 1.0, 2.0), "cpu", 0);
    Variable x = new Variable(xTensor, false);
    Variable y = new Variable(yTensor, false);
    SGD opt = new SGD(linear.parameters(), 0.05);

    Variable pred0 = linear.forward(x);
    Variable loss0 = lognn_java.mse_loss(pred0, y);
    double before = scalar(loss0);

    opt.zero_grad();
    loss0.backward();
    opt.step();

    Variable pred1 = linear.forward(x);
    Variable loss1 = lognn_java.mse_loss(pred1, y);
    double after = scalar(loss1);
    double grad0 = new DoubleVector(linear.parameters().get(0).grad().get_data()).get(0);
    if (Double.isNaN(grad0)) throw new RuntimeException("grad parity failed");

    if (Double.isNaN(before) || Double.isNaN(after) || Math.abs(before - after) < 1e-12) {
      throw new RuntimeException("Java SWIG smoke failed: before=" + before + ", after=" + after);
    }

    // Module constructor coverage parity.
    ModulePtrVector mods = new ModulePtrVector();
    Sequential seq = new Sequential(mods);
    seq.parameters();
    new Conv2d(1, 2, 3, 3, 1, 1, 1, 1, true, "cpu", 0);
    new MaxPool2d(2, 2, 0);
    new AvgPool2d(2, 2, 0);
    new ConvTranspose2d(2, 1, 3, 3, 1, 1, 1, 1, 0, 0, true, "cpu", 0);
    new BatchNorm2d(2, 0.1, 1e-5, "cpu", 0);
    new TransformerEncoderLayer(4, 0.1, "cpu", 0);
    new TransformerEncoder(1, 4, 0.1, "cpu", 0);
    new CausalTransformerEncoderLayer(4, 0.1, "cpu", 0);
    new CausalTransformerEncoder(1, 4, 0.1, "cpu", 0);

    // Checkpoint + diagnostics parity.
    String ckpt = System.getProperty("java.io.tmpdir") + "/lognn_java_parity.bin";
    lognn_java.save_model(new TransformerEncoder(1, 4, 0.1, "cpu", 0), ckpt);
    lognn_java.load_model(new TransformerEncoder(1, 4, 0.1, "cpu", 0), ckpt);
    lognn_java.is_mlx_native_enabled();
    lognn_java.mlx_dispatch_count();
    lognn_java.reset_mlx_dispatch_count();

    System.out.printf("Java SWIG parity smoke passed. Loss before=%.6f, after=%.6f%n", before, after);
  }
}
