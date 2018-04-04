import com.google.common.math.DoubleMath;
import org.javatuples.Quartet;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.dimensionalityreduction.PCA;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.Arrays;

public class FastKde {
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

    if(gridSize == 200) {
      gridSize = Math.max(x.length(), 512);
    }
    gridSize = (int) Math.pow(2, Math.ceil(DoubleMath.log2(gridSize)));

    Quartet<Double, Double, Double, Double> extents = new Quartet<>(
            x.minNumber().doubleValue(),
            x.maxNumber().doubleValue(),
            y.minNumber().doubleValue(),
            y.maxNumber().doubleValue());

    double deltaX = (extents.getValue1() - extents.getValue0()) / (gridSize - 1);
    double deltaY = (extents.getValue3() - extents.getValue2()) / (gridSize - 1);

    INDArray bins = Nd4j.vstack(x, y).transpose();
    bins.subiRowVector(Nd4j.create(new double[] {extents.getValue0(), extents.getValue2()}));
    bins.diviRowVector(Nd4j.create(new double[] {deltaX, deltaY}));
    bins = Transforms.floor(bins).transpose();

    //2d histogram of x and y
    //TODO : GRID = COO_MATRIX

    INDArray[] covarianceMatrix = PCA.covarianceMatrix(bins);


    return null;
  }

  public static INDArray fastKde2d(INDArray x, INDArray y) {
    x = Nd4j.stripOnes(x);
    y = Nd4j.stripOnes(y);
    INDArray weights = Nd4j.create(x.length()).add(1);

    return fastKde2d(x, y, 200, false, weights, 1.0);
  }

  public static void main(String[] args) {
    INDArray x = Nd4j.create(new double[] {1, 2, 3});
    INDArray y = Nd4j.create(new double[] {4, 5, 6});
    INDArray bins = Nd4j.vstack(x, y).transpose();
    System.out.println(bins);
    System.out.println("----");
    bins.subiRowVector(Nd4j.create(new double[] {1, 4}));
    System.out.println(bins);
    System.out.println("----");
    bins.diviRowVector(Nd4j.create(new double[] {0.5, 0.5}));
    System.out.println(bins);
    System.out.println("Bins Array ----");
    bins = Transforms.floor(bins).transpose();
    System.out.println(bins);
    System.out.println("Covariance Matrix ------");
    INDArray[] covarianceMatrix = PCA.covarianceMatrix(bins);
    System.out.println(Arrays.deepToString(covarianceMatrix));
  }
}

