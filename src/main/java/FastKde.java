import org.javatuples.Pair;
import org.javatuples.Quartet;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.NioUtil;
import sun.rmi.rmic.iiop.IDLNames;

import java.util.Arrays;

public class FastKde {

  public static INDArray fastKde2d(
          INDArray x,
          INDArray y,
          int gridSize,
          Boolean noCorrelation,
          INDArray weights,
          Double adjust) {

    //--------------------------------------- Input handling -------------------------------------------------------
    x = Nd4j.stripOnes(x);
    y = Nd4j.stripOnes(y);
    weights = Nd4j.stripOnes(weights);

    if (x.length() != y.length()) {
      throw new IllegalArgumentException(
              "INDArrays x and y don't have the same length.");
    }

    if (weights.length() != x.length()) {
      throw new IllegalArgumentException(
              "INDArray weights doesn't have the same length as x and y.");
    }

    //---------------------------------------- Optimize grid size --------------------------------------------------

    gridSize = Utilities.optimizeGridSize(gridSize, x.length()) + 2;
    System.out.println("Gridsize: " + gridSize);

    //---------------------------------------- 2d histogram --------------------------------------------------------

    Quartet<Double, Double, Double, Double> extents = new Quartet<>(
            x.minNumber().doubleValue(),
            x.maxNumber().doubleValue(),
            y.minNumber().doubleValue(),
            y.maxNumber().doubleValue());

    System.out.println("xMin: " + extents.getValue0() + ", xMax: " + extents.getValue1() + ", yMin: " + extents.getValue2() + ", yMax: " + extents.getValue3());

    double deltaX = (extents.getValue1() - extents.getValue0()) / (gridSize - 1);
    double deltaY = (extents.getValue3() - extents.getValue2()) / (gridSize - 1);

    INDArray bins = Utilities.getBins(x, y, deltaX, deltaY, extents);

    System.out.println("bins: ");
    System.out.println(bins);

    INDArray grid = Utilities.getCooMatrix(weights, bins, gridSize);
    //System.out.println("Grid: (" + grid.rows() + ", " + grid.columns() + ")");

    //------------------------------------------ Kernel preliminary calculations -----------------------------------

    INDArray covariance = Utilities.getCovariance(bins, noCorrelation);
    double scottsFactor = Utilities.getScottsFactor(x.length(), adjust);
    System.out.println("Covariance:");
    System.out.println(covariance);
    System.out.println("Scott's Factor: " + scottsFactor);

    //------------------------------------------ Make the Gaussian kernel ------------------------------------------

    INDArray standardDeviations = Utilities.getStandardDeviations(covariance);
    System.out.println("Standard Deviation: ");
    System.out.println(standardDeviations);

    INDArray kernN = Utilities.getKernN(standardDeviations, scottsFactor);
    double kernNx = kernN.getDouble(0, 0);
    double kernNy = kernN.getDouble(0, 1);
    System.out.println("kern_N: " + kernN);

    INDArray bandwidth = Utilities.getBandwidth(covariance, scottsFactor);
    System.out.println("Bandwidth:");
    System.out.println(bandwidth);

    INDArray xCoords = Nd4j.arange(kernNx).sub(kernNx / 2.0);
    INDArray yCoords = Nd4j.arange(kernNy).sub(kernNy / 2.0);
    Pair<INDArray, INDArray> meshGrid = Utilities.getMeshGrid(xCoords, yCoords);
    System.out.println("Meshgrid: (" + meshGrid.getValue0().rows() + ", " + meshGrid.getValue0().columns() + "), (" + meshGrid.getValue1().rows() + ", " + meshGrid.getValue1().columns() + ")" );

     INDArray kernel = Utilities.getKernel(kernNx, kernNy,
            meshGrid.getValue0(), meshGrid.getValue1(), bandwidth);
    System.out.println("Kernel:");
    System.out.println("(" + kernel.rows() + ", " + kernel.columns() + ")");

    //------------------------------------------ Produce the kernel density estimate -------------------------------

    System.out.println("Grid: (" + grid.rows() + ", " + grid.columns() + ")");

    //TODO : convolveGrid Tests, this might not work as desired
    grid = Utilities.convolveGrid(grid, kernel);
//    System.out.println("Grid Before Norm:");
//    System.out.println("(" + grid.rows() + ", " + grid.columns() + ")");

    INDArray normalizationFactor = Utilities.getNormalizationFactor(covariance,
            scottsFactor, x.length(), deltaX, deltaY);
    //System.out.println("Normalization Factor: " + normalizationFactor);

    grid = grid.div(normalizationFactor);
    //grid = grid.diviRowVector(normalizationFactor);
//    System.out.println("Grid After Norm:");
    //System.out.println(Arrays.toString(grid.shape()));
    grid = grid.reshape(gridSize, gridSize);
    //System.out.println("(" + grid.rows() + ", " + grid.columns() + ")");
//    System.out.println(grid);
    return grid;
  }

  public static INDArray fastKde2d(INDArray x, INDArray y) {
    x = Nd4j.stripOnes(x);
    y = Nd4j.stripOnes(y);
    INDArray weights = Nd4j.create(x.length()).add(1);

    return fastKde2d(x, y, 200, false, weights, 1.0);
  }
}

