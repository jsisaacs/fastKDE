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

    gridSize = Utilities.optimizeGridSize(gridSize, x.length());

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
    INDArray grid = Nd4j.zeros(gridSize, gridSize);
    System.out.println("grid");
    System.out.println(grid);
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

    INDArray kernel = Utilities.getKernel(kernNx, kernNy,
            meshGrid.getValue0(), meshGrid.getValue1(), bandwidth);

    //------------------------------------------ Produce the kernel density estimate -------------------------------

    //TODO : convolveGrid Tests, this might not work as desired
    grid = Utilities.convolveGrid(grid, kernel);

    INDArray normalizationFactor = Utilities.getNormalizationFactor(covariance,
            scottsFactor, x.length(), deltaX, deltaY);

    grid.diviRowVector(normalizationFactor);

    return grid;
  }

  public static INDArray fastKde2d(INDArray x, INDArray y) {
    x = Nd4j.stripOnes(x);
    y = Nd4j.stripOnes(y);
    INDArray weights = Nd4j.create(x.length()).add(1);

    return fastKde2d(x, y, 200, false, weights, 1.0);
  }

  public static void main(String[] args) {
//    INDArray x = Nd4j.create(new double[] {1, 2, 3});
//    INDArray y = Nd4j.create(new double[] {4, 5, 6});
//    INDArray bins = Nd4j.vstack(x, y).transpose();
//    System.out.println(bins);
//    System.out.println("----");
//    bins.subiRowVector(Nd4j.create(new double[] {1, 4}));
//    System.out.println(bins);
//    System.out.println("----");
//    bins.diviRowVector(Nd4j.create(new double[] {1, 1}));
//    System.out.println(bins);
//    System.out.println("Bins Array ----");
//    bins = Transforms.floor(bins).transpose();
//    System.out.println(bins);
//    System.out.println("Covariance Matrix ------");
//    INDArray[] covarianceMatrix = PCA.covarianceMatrix(bins);
//    INDArray[] covmean = PCA.covarianceMatrix(x);
//    INDArray cov = covmean[0];
//    System.out.println(cov);

//    double[][] matrix = new double[][] {
//            {1., 1.,},
//            {1., 1.}
//            };
//    INDArray A = Nd4j.create(matrix);
//    System.out.println("A ----");
//    System.out.println(A);
//    System.out.println("2 ----");
//    //System.out.println(A.getDouble(0, 2));
//    //INDArray[] covarianceMatrix = PCA.covarianceMatrix(a);
//    //System.out.println(covarianceMatrix[0]);
//    INDArray[] covMatrix = PCA.covarianceMatrix(A);
//
//    System.out.println("Cov ----");
//    System.out.println(covMatrix[0]);
//    System.out.println("StdDev Test ------");

//    System.out.println(standardDeviations);
//    System.out.println(0.8326831776556043 * 2 * BigDecimalMath.PI.doubleValue());
//    INDArray testArray = Transforms.round(standardDeviations.mul(0.8326831776556043 * 2 * BigDecimalMath.PI.doubleValue()));
//    System.out.println(testArray);
//    double[][] matrix = new double[][] {
//            {6., 4.,},
//            {2., 8.}
//    };
//    INDArray A = Nd4j.create(matrix);
//    double scottsFactor = 0.823;
//    INDArray standardDeviations = Nd4j.diag(Transforms.sqrt(A));
//    INDArray kern_n = Transforms.round(standardDeviations.mul(scottsFactor * 2 * BigDecimalMath.PI.doubleValue()));
//    double kern_nx = kern_n.getDouble(0, 0);
//    double kern_ny = kern_n.getDouble(1, 0);
//    System.out.println(kern_nx);
//    System.out.println(kern_ny);
//    INDArray xCoords = Nd4j.arange(kern_nx).sub(kern_nx / 2.0);
//    System.out.println(xCoords);
//    INDArray yCoords = Nd4j.arange(kern_ny).sub(kern_ny / 2.0);
//    System.out.println(yCoords);
//    System.out.println("________");
//    INDArray grid = Nd4j.zeros(4, 4);
//    System.out.println(grid);
    //TEST FOR MESH GRID BELOW
//    double[] matrix1 = new double[]
//            {2.,2.};
//    double[] matrix2 = new double[]
//            {1.,3.};
//    INDArray x = Nd4j.create(matrix1);
//    INDArray y = Nd4j.create(matrix2);
    //Pair<INDArray, INDArray> pair = getMeshGrid(x, y);
    //INDArray kernel = Nd4j.vstack(Nd4j.toFlattened(pair.getValue0()),
    //        Nd4j.toFlattened(pair.getValue1()));
    //kernel =
    //System.out.println(kernel);
    //double[][] matrix3 = new double[][] {
    //        {3.,-2.},
    //        {-1.,2.}
    //};
    //INDArray invCov = Nd4j.create(matrix3);
    //System.out.println(invCov);
    //Dot dot = new Dot(invCov, invCov);
    //System.out.println(dot);
  }
}

