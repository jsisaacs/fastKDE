import com.google.common.math.DoubleMath;
import org.javatuples.Pair;
import org.javatuples.Quartet;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dimensionalityreduction.PCA;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.inverse.InvertMatrix;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.linalg.util.BigDecimalMath;

public class FastKde {

  private static int optimizeGridSize(int gridSize, int xLength) {
    if(gridSize == 200) {
      gridSize = Math.max(xLength, 512);
    }
    gridSize = (int) Math.pow(2, Math.ceil(DoubleMath.log2(gridSize)));
    return gridSize;
  }

  private static INDArray getBins(INDArray x,
                                  INDArray y,
                                  double deltaX,
                                  double deltaY,
                                  Quartet<Double, Double, Double, Double> extents) {
    INDArray bins = Nd4j.vstack(x, y).transpose();
    bins.subiRowVector(Nd4j.create(new double[] {extents.getValue0(), extents.getValue2()}));
    bins.diviRowVector(Nd4j.create(new double[] {deltaX, deltaY}));
    bins = Transforms.floor(bins).transpose();
    return bins;
  }

  private static INDArray getCovariance(INDArray bins) {
    INDArray[] covarianceMatrix = PCA.covarianceMatrix(bins);
    INDArray covariance = covarianceMatrix[0];
    return covariance;
  }

  private static Pair<INDArray, INDArray> getMeshGrid(INDArray x, INDArray y) {
    int numRows = y.length();
    int numCols = x.length();

    x = x.reshape(1, numCols);
    y = y.reshape(numRows, 1);

    INDArray X = x.repeat(0, numRows);
    INDArray Y = y.repeat(1, numCols);

    return new Pair<>(X, Y);
  }

  private static double getScottsFactor(INDArray x, double adjust) {
    return Math.pow(x.length(), (-1. / 6.)) * adjust;
  }

  private static INDArray getBandwidth(INDArray x, INDArray covariance, double adjust) {
    double scottsFactor = Math.pow(x.length(), (-1. / 6.)) * adjust;
    return InvertMatrix.invert(covariance.mul(Math.pow(scottsFactor, 2)), true);
  }

  private static INDArray getStandardDeviations(INDArray covariance) {
    return Nd4j.diag(Transforms.sqrt(covariance));
  }

  private static INDArray getKernel(Pair<INDArray, INDArray> meshGrid,
                                    INDArray bandwidth,
                                    INDArray standardDeviations,
                                    double scottsFactor) {
    INDArray kernel = Nd4j.vstack(Nd4j.toFlattened(meshGrid.getValue0()),
            Nd4j.toFlattened(meshGrid.getValue1()));
    System.out.println(kernel);
    return null;
  }

//  INDArray kernel = Nd4j.vstack(Nd4j.toFlattened(meshGrid.getValue0()),
//          Nd4j.toFlattened(meshGrid.getValue1()));
//    System.out.println(kernel);
//    return null;

  //private INDArray getKernel()


  public static INDArray fastKde2d(
          INDArray x,
          INDArray y,
          int gridSize,
          Boolean noCorrelation,
          INDArray weights,
          Double adjust) {
    x = Nd4j.stripOnes(x);
    y = Nd4j.stripOnes(y);
    weights = Nd4j.stripOnes(weights);

    if (x.length() != y.length()) {
      throw new IllegalArgumentException("INDArrays x and y don't have the same length.");
    }

    if (weights.length() != x.length()) {
      throw new IllegalArgumentException("INDArray weights doesn't have the same length as x and y.");
    }

    gridSize = optimizeGridSize(gridSize, x.length());

    Quartet<Double, Double, Double, Double> extents = new Quartet<>(
            x.minNumber().doubleValue(),
            x.maxNumber().doubleValue(),
            y.minNumber().doubleValue(),
            y.maxNumber().doubleValue());

    double deltaX = (extents.getValue1() - extents.getValue0()) / (gridSize - 1);
    double deltaY = (extents.getValue3() - extents.getValue2()) / (gridSize - 1);

    INDArray bins = getBins(x, y, deltaX, deltaY, extents);

    //TODO : 2d histogram of x and y
    INDArray grid = Nd4j.zeros(gridSize, gridSize);

    //TODO : THIS DOESN'T WORK WELL
    INDArray covariance = getCovariance(bins);

    if (noCorrelation) {
      covariance.putScalar(new int[] {1,0}, 0.0);
      covariance.putScalar(new int[] {0,1}, 0.0);
    }

    double scottsFactor = Math.pow(x.length(), (-1. / 6.)) * adjust;

    INDArray standardDeviations = Nd4j.diag(Transforms.sqrt(covariance));
    INDArray kern_n = Transforms.round(standardDeviations.mul(scottsFactor * 2 * BigDecimalMath.PI.doubleValue()));
    double kern_nx = kern_n.getDouble(0, 0);
    double kern_ny = kern_n.getDouble(1, 0);

    INDArray inverseCovariance = InvertMatrix.invert(covariance.mul(Math.pow(0.5, 2)), true);

    //mesh grid
    INDArray xCoords = Nd4j.arange(kern_nx).sub(kern_nx / 2.0);
    INDArray yCoords = Nd4j.arange(kern_ny).sub(kern_ny / 2.0);
    Pair<INDArray, INDArray> meshGrid = getMeshGrid(xCoords, yCoords);

    //System.out.println(meshGrid.getValue0());
    //System.out.println(Nd4j.toFlattened(meshGrid.getValue0()));
    //System.out.println(Nd4j.toFlattened(meshGrid.getValue1()));

    return null;
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
    double[] matrix1 = new double[]
            {1.,2.,3.};
    double[] matrix2 = new double[]
            {4.,5.,6.};
    INDArray x = Nd4j.create(matrix1);
    INDArray y = Nd4j.create(matrix2);
    Pair<INDArray, INDArray> pair = getMeshGrid(x, y);
    INDArray kernel = Nd4j.vstack(Nd4j.toFlattened(pair.getValue0()),
            Nd4j.toFlattened(pair.getValue1()));
    //kernel =
    //System.out.println(kernel);

  }
}

