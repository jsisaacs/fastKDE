import org.javatuples.Pair;
import org.javatuples.Quartet;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.CustomOp;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.impl.layers.convolution.Conv2D;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Conv2DConfig;
import org.nd4j.linalg.cpu.nativecpu.NDArray;
import org.nd4j.linalg.dimensionalityreduction.PCA;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.convolution.Convolution;

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
    assertEquals(covariance, Utilities.getCovariance(Utilities.getBins(x, y, 0.011764705882352941, 0.011764705882352941, extents), false));
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
//    INDArray grid = Nd4j.ones(3, 3);
//    INDArray kernel = Nd4j.ones(4, 4);
//    INDArray output = Nd4j.zeros(6, 6);
//
//    assertEquals(output, Utilities.convolveGrid(grid, kernel));
    INDArray input = Nd4j.create(new double[][]{
            {3, 2, 5, 6, 7, 8},
            {5, 4, 2, 10, 8, 1}
    });
    INDArray kernel = Nd4j.create(new double[][]{
            {4, 5},
            {1, 2}
    });
    INDArray output = Nd4j.ones(input.rows(), input.columns());

    CustomOp convolve2d = Conv2D.builder()
            .sameDiff()
            .inputFunctions()
            .inputArrays()
            .outputs(output)
            .config(config)
            .build();
    Nd4j.getExecutioner().exec(convolve2d);

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
