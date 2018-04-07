import org.javatuples.Quartet;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static junit.framework.Assert.assertEquals;

public class FastKdeTestSuite {

//  @Test
//  public void optimizeGridSizeTest() {
//
//    //small grid
//    int gridSize= 5;
//    int xLength = 20;
//    assertEquals(8, FastKde.optimizeGridSize(gridSize, xLength));
//
//    //default grid
//    gridSize = 200;
//    xLength = 200;
//    assertEquals(512, FastKde.optimizeGridSize(gridSize, xLength));
//
//    //large grid
//    gridSize = 1000;
//    xLength = 1000000;
//    assertEquals(1024, FastKde.optimizeGridSize(gridSize, xLength));
//  }

//  @Test
//  public void getBinsTest() {
//
//    //small inputs
//    INDArray x = Nd4j.create(new double[] {1., 2., 3., 4.});
//    INDArray y = Nd4j.create(new double[] {5., 6., 7., 8.});
//    INDArray bins = Nd4j.create(new double[][] {
//            {0., 85., 170., 255.},
//            {0., 85., 170., 255.}
//    });
//    Quartet<Double, Double, Double, Double> extents = new Quartet<>(
//            x.minNumber().doubleValue(),
//            x.maxNumber().doubleValue(),
//            y.minNumber().doubleValue(),
//            y.maxNumber().doubleValue());
//    assertEquals(bins, FastKde.getBins(x, y, 0.011764705882352941, 0.011764705882352941, extents));
//
//    //large inputs
//    //TODO
//  }

  //@Test
  public void getCovarianceTest() {

  }

  //@Test
  public void getMeshGridTest() {

  }

  //@Test
  public void getScottsFactorTest() {

  }

  //@Test
  public void getBandwidthTest() {

  }

  //@Test
  public void getStandardDeviationsTest() {

  }

  //@Test
  public void getKernelTest() {

  }

}
