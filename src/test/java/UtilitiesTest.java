import org.javatuples.Quartet;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static junit.framework.Assert.assertEquals;

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
            {12041.66666667, 12041.66666667},
            {12041.66666667, 12041.66666667}
    });
    Quartet<Double, Double, Double, Double> extents = new Quartet<>(
            x.minNumber().doubleValue(),
            x.maxNumber().doubleValue(),
            y.minNumber().doubleValue(),
            y.maxNumber().doubleValue());
    assertEquals(covariance, Utilities.getCovariance(Utilities.getBins(x, y, 0.011764705882352941, 0.011764705882352941, extents), false));
  }

  @Test
  public void getScottsFactorTest() {
    INDArray x = Nd4j.create(new double[] {1., 2., 3., 4.});
    double adjust = 1.;

    assertEquals(0.7937005259840998, Utilities.getScottsFactor(x.length(), adjust));
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

  //@Test
  public void getBandwidthTest() {
    //TODO
  }

  //@Test
  public void getMeshGridTest() {
    //TODO
  }

  //@Test
  public void getKernelTest() {
    //TODO
  }

}
