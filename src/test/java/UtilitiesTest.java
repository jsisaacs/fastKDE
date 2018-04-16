import org.javatuples.Pair;
import org.javatuples.Quartet;
import org.junit.Test;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Conv2DConfig;
import org.nd4j.linalg.factory.Nd4j;

import java.awt.image.Kernel;

import static org.junit.Assert.assertEquals;

public class UtilitiesTest {
  @Test
  public void optimizeGridSizeTest() {
    //small grid
    int gridSize= 5;
    int xLength = 20;
    assertEquals(8, Utilities.optimizeGridSize(gridSize, xLength));

    //default grid
    gridSize = 200;
    xLength = 200;
    assertEquals(512, Utilities.optimizeGridSize(gridSize, xLength));

    //large grid
    gridSize = 1000;
    xLength = 1000000;
    assertEquals(1024, Utilities.optimizeGridSize(gridSize, xLength));
  }

  @Test
  public void getBinsTest() {
    //small inputs
    INDArray x = Nd4j.create(new double[] {1., 2., 3., 4.});
    INDArray y = Nd4j.create(new double[] {5., 6., 7., 8.});
    INDArray bins = Nd4j.create(new double[][] {
            {0., 85., 170., 255.},
            {0., 85., 170., 255.}
    });
    Quartet<Double, Double, Double, Double> extents = new Quartet<>(
            x.minNumber().doubleValue(),
            x.maxNumber().doubleValue(),
            y.minNumber().doubleValue(),
            y.maxNumber().doubleValue());
    assertEquals(bins, Utilities.getBins(x, y, 0.011764705882352941, 0.011764705882352941, extents));

    //large inputs
    //TODO
  }

  @Test
  public void gridTest() {
    INDArray kernel = Nd4j.ones(13, 13);
    kernel = kernel.mul(4);
    INDArray grid = Nd4j.ones(514, 514);
    grid = grid.mul(3);
    System.out.println(Utilities.convolveGrid(grid, kernel));
  }

  @Test
  public void getCOOMatrixTest() {

  }

  //TODO
  @Test
  public void getCovarianceTest() {
    INDArray x = Nd4j.create(new double[] {1., 2., 3., 4.});
    INDArray y = Nd4j.create(new double[] {5., 6., 7., 8.});
    INDArray covariance = Nd4j.create(new double[][] {
            {2.5288, 2.5288},
            {2.5288, 2.5288}
    });
    Quartet<Double, Double, Double, Double> extents = new Quartet<>(
            x.minNumber().doubleValue(),
            x.maxNumber().doubleValue(),
            y.minNumber().doubleValue(),
            y.maxNumber().doubleValue());

    System.out.println(Utilities.getBins(x, y, 0.011764705882352941, 0.011764705882352941, extents));
    INDArray covarianceOutput = Utilities.getCovariance(Utilities.getBins(x, y, 0.011764705882352941, 0.011764705882352941, extents), false);
    System.out.println(covarianceOutput);
    assertEquals(covariance.getDouble(0, 0), covarianceOutput.getDouble(0, 0), 1);

    //assertEquals(covariance, Utilities.getCovariance(Utilities.getBins(x, y, 0.011764705882352941, 0.011764705882352941, extents), false));
  }

  @Test
  public void getScottsFactorTest() {
    INDArray x = Nd4j.create(new double[] {1., 2., 3., 4.});
    double adjust = 1.;
    assertEquals(0.7937005259840998, Utilities.getScottsFactor(x.length(), adjust), 0.00001);
  }

  @Test
  public void getStandardDeviationsTest() {
    INDArray covariance = Nd4j.create(new double[][] {
            {12041.66666667, 12041.66666667},
            {12041.66666667, 12041.66666667}
    });
    INDArray standardDeviations = Nd4j.create(new double[] {109.73452814, 109.73452814});
    assertEquals(standardDeviations, Utilities.getStandardDeviations(covariance));
  }

  @Test
  public void diagTest() {
    SameDiff sd = SameDiff.create();

    INDArray ia = Nd4j.create(new float[]{4, 2});
    SDVariable in = sd.var("in", new int[]{1, 2});
    INDArray expOut = Nd4j.create(new int[]{2, 2});
    DynamicCustomOp diag = DynamicCustomOp.builder("diag").addInputs(ia).addOutputs(expOut).build();
    Nd4j.getExecutioner().exec(diag);
    SDVariable t = sd.diag(in);

    SDVariable loss = sd.max("loss", t, 0, 1);

    sd.associateArrayWithVariable(ia, in);
    sd.exec();
    INDArray out = t.getArr();
    System.out.println(out);
  }

  @Test
  public void getKernNTest() {
    INDArray kernN = Nd4j.create(new double[] {547., 547.});
    INDArray standardDeviations = Nd4j.create(new double[] {109.73452814, 109.73452814});
    double scottsFactor = 0.7937005259840998;
    assertEquals(kernN, Utilities.getKernN(standardDeviations, scottsFactor));
  }

  @Test
  public void getBandwidthTest() {
    INDArray covariance = Nd4j.create(new double[][] {
            {12., 13.},
            {14., 15.}
    });
    double scottsFactor = 0.75;
    INDArray bandwidth = Nd4j.create(new double[][] {
            {-13.3333, 11.5556},
            {12.4444, -10.6667}
    });
    assertEquals(bandwidth, Utilities.getBandwidth(covariance, scottsFactor));
  }

  @Test
  public void getMeshGridTest() {
    int nx = 3;
    int ny = 2;
    INDArray x = Nd4j.linspace(0, 1, nx);
    INDArray y = Nd4j.linspace(0, 1, ny);
    INDArray X = Nd4j.create(new double[][] {
            {0., 0.5, 1.},
            {0., 0.5, 1.}
    });
    INDArray Y = Nd4j.create(new double[][] {
            {0., 0., 0.},
            {1., 1., 1.}
    });

    Pair<INDArray, INDArray> meshGrid = Utilities.getMeshGrid(x, y);
    assertEquals(X, meshGrid.getValue0());
    assertEquals(Y, meshGrid.getValue1());
  }

  @Test
  public void getKernelTest() {
    INDArray X = Nd4j.create(new double[][] {
            {0., 0.5, 1.},
            {0., 0.5, 1.}
    });
    INDArray Y = Nd4j.create(new double[][] {
            {0., 0., 0.},
            {1., 1., 1.}
    });
    INDArray bandwidth = Nd4j.create(new double[][] {
            {-13.33, 11.56},
            {12.44, -10.67}
    });
    INDArray kernN = Nd4j.create(new double[] {2., 3.});
    double kernNx = kernN.getDouble(0, 0);
    double kernNy = kernN.getDouble(0, 1);
    INDArray kernel = Nd4j.create(new double[][] {
            {1., 5.2923},
            {784.4634, 207.4728},
            {2.7217, 1.}
    });
    assertEquals(kernel, Utilities.getKernel(kernNx, kernNy, X, Y, bandwidth));
  }

  @Test
  public void convolveGridTest() {
    int nIn = 1;
    int nOut = 1;
    int kH = 4;
    int kW = 4;

    int mb = 1;
    int imgH = 12;
    int imgW = 12;

    SameDiff sd = SameDiff.create();
    INDArray wArr = Nd4j.ones(nOut, nIn, kH, kW); //As per DL4J
    wArr = wArr.mul(4);
    System.out.println(wArr);
    System.out.println("==--===");
    INDArray bArr = Nd4j.ones(1, nOut);
    System.out.println(bArr);
    System.out.println("-------");
    INDArray inArr = Nd4j.ones(mb, nIn, imgH, imgW);
    inArr = inArr.mul(3);
    System.out.println(inArr);
    System.out.println("--------");

    SDVariable in = sd.var("in", inArr);
    SDVariable w = sd.var("W", wArr);
    SDVariable b = sd.var("b", bArr);

    //Order: https://github.com/deeplearning4j/libnd4j/blob/6c41ea5528bb1f454e92a9da971de87b93ff521f/include/ops/declarable/generic/convo/conv2d.cpp#L20-L22
    //in, w, b - bias is optional
    SDVariable[] vars = new SDVariable[]{in, w, b};

    Conv2DConfig c = Conv2DConfig.builder()
            .kh(kH).kw(kW)
            .ph(1).pw(1)
            .sy(1).sx(1)
            .dh(1).dw(1)
            .isSameMode(true)
            .build();
    sd.conv2d(vars, c);
    INDArray outArr = sd.execAndEndResult();

    INDArray grid = Nd4j.ones(12, 12);
    grid = grid.mul(3);
    INDArray kernel = Nd4j.ones(4, 4);
    kernel = kernel.mul(4);
    assertEquals(outArr, Utilities.convolveGrid(grid, kernel));
  }

  @Test
  public void normalizationFactorTest() {
    INDArray covariance = Nd4j.create(new double[][] {
            {12., 13.},
            {14., 15.}
    });
    double scottsFactor = 0.75;
    System.out.println(Utilities.getNormalizationFactor(covariance, scottsFactor, 4, 0.5, 0.5));

  }
}
