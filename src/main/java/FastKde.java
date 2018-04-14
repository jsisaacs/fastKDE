import org.javatuples.Pair;
import org.javatuples.Quartet;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

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

    //gridSize = Utilities.optimizeGridSize(gridSize, x.length());
    gridSize = 514;

    //---------------------------------------- 2d histogram --------------------------------------------------------

    Quartet<Double, Double, Double, Double> extents = new Quartet<>(
            x.minNumber().doubleValue(),
            x.maxNumber().doubleValue(),
            y.minNumber().doubleValue(),
            y.maxNumber().doubleValue());

    double deltaX = (extents.getValue1() - extents.getValue0()) / (gridSize - 1);
    double deltaY = (extents.getValue3() - extents.getValue2()) / (gridSize - 1);

    INDArray bins = Utilities.getBins(x, y, deltaX, deltaY, extents);
    System.out.println("Bins");
    System.out.println(bins);

    // TODO : implement the sparse matrix, figure out indices
    INDArray grid = Nd4j.ones(gridSize, gridSize);
    System.out.println("grid");
    System.out.println(grid);
    System.out.println(grid.rows() + ", " + grid.columns());
    //INDArray grid = Nd4j.createSparseCOO( , gridIndices, new int[] {gridSize, gridSize});

    //------------------------------------------ Kernel preliminary calculations -----------------------------------

    INDArray covariance = Utilities.getCovariance(bins, noCorrelation);
    double scottsFactor = Utilities.getScottsFactor(x.length(), adjust);
    System.out.println("Cov");
    System.out.println(covariance);
    System.out.println("scotts factor: " + scottsFactor);

    //------------------------------------------ Make the Gaussian kernel ------------------------------------------

    INDArray standardDeviations = Utilities.getStandardDeviations(covariance);
    System.out.println("stdDevs");
    System.out.println(standardDeviations);

    INDArray kernN = Utilities.getKernN(standardDeviations, scottsFactor);
    double kernNx = kernN.getDouble(0, 0);
    double kernNy = kernN.getDouble(1, 0);
    System.out.println("kernNx: " + kernNx);
    System.out.println("kernNy: " + kernNy);

    INDArray bandwidth = Utilities.getBandwidth(covariance, scottsFactor);
    System.out.println("Bandwidth");
    System.out.println(bandwidth);

    INDArray xCoords = Nd4j.arange(kernNx).sub(kernNx / 2.0);
    INDArray yCoords = Nd4j.arange(kernNy).sub(kernNy / 2.0);
    Pair<INDArray, INDArray> meshGrid = Utilities.getMeshGrid(xCoords, yCoords);
    System.out.println(meshGrid.getValue0().rows() + ", " + meshGrid.getValue0().columns());
    System.out.println(meshGrid.getValue1().rows() + ", " + meshGrid.getValue1().columns());
    System.out.println(meshGrid.getValue0());
    System.out.println("---------------------------------");
    System.out.println(meshGrid.getValue1());

     INDArray kernel = Utilities.getKernel(kernNx, kernNy,
            meshGrid.getValue0(), meshGrid.getValue1(), bandwidth);
    System.out.println("Kernel Done");

    System.out.println(kernel.rows() + ", " + kernel.columns());
    System.out.println(gridSize);

    //------------------------------------------ Produce the kernel density estimate -------------------------------

    //TODO : convolveGrid Tests, this might not work as desired
    grid = Utilities.convolveGrid(grid, kernel);
    System.out.println("Grid Done");
    System.out.println(grid);

    INDArray normalizationFactor = Utilities.getNormalizationFactor(covariance,
            scottsFactor, x.length(), deltaX, deltaY);
    System.out.println("Normalization Calculatated");
    System.out.println(normalizationFactor);

    grid.diviRowVector(normalizationFactor);
    System.out.println("Normalization Applied");

    System.out.println(grid);
    return grid;
  }

  public static INDArray fastKde2d(INDArray x, INDArray y) {
    x = Nd4j.stripOnes(x);
    y = Nd4j.stripOnes(y);
    INDArray weights = Nd4j.create(x.length()).add(1);

    return fastKde2d(x, y, 200, false, weights, 1.0);
  }
}

