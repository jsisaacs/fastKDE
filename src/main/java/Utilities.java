import com.google.common.math.DoubleMath;
import org.apache.commons.math3.linear.RealMatrix;
import org.javatuples.Pair;
import org.javatuples.Quartet;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Conv2DConfig;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.inverse.InvertMatrix;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.linalg.util.BigDecimalMath;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.CustomOp;
import org.apache.commons.math3.stat.correlation.Covariance;

import java.util.Arrays;

/*
The MIT License (MIT)

Copyright (c) 2018 Joshua Isaacson, Morgan Fouesneau & contributors

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

public class Utilities {
  public static int optimizeGridSize(int gridSize,
                                     int xLength) {
    if (gridSize <= 0 || xLength <= 0) {
      throw new IllegalArgumentException("Arguments must be greater than 0.");
    }
    if(gridSize == 1) {
      gridSize = Math.max(xLength, 512);
    }
    gridSize = (int) Math.pow(2, Math.ceil(DoubleMath.log2(gridSize)));
    return gridSize;
  }

  public static INDArray getBins(INDArray x,
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

  //TODO
  public static INDArray getCooMatrix(INDArray weights, INDArray bins, int gridSize) {
    double[] weightsArray = new double[weights.columns()];

    for (int i = 0; i < weights.rows(); i++) {
      weightsArray[i] = weights.getDouble(i);
    }

    int[][] indices = bins.toIntMatrix();

    int[] shape = new int[] {gridSize, gridSize};

    return Nd4j.createSparseCOO(weightsArray, indices, shape).toDense();
  }

  public static INDArray getCovariance(INDArray bins,
                                       boolean noCorrelation) {
    bins = bins.transpose();

    double[][] binsArray = bins.toDoubleMatrix();

    Covariance apacheCommonsCovariance = new Covariance(binsArray);
    RealMatrix rm = apacheCommonsCovariance.getCovarianceMatrix();
    rm.getData();
    INDArray covariance = Nd4j.create(rm.getData());

    if (noCorrelation) {
      covariance.putScalar(new int[] {1,0}, 0.0);
      covariance.putScalar(new int[] {0,1}, 0.0);
    }

    return covariance.div(10000);
  }

  public static double getScottsFactor(int xLength,
                                       double adjust) {
    return Math.pow(xLength, (-1. / 6.)) * adjust;
  }

  public static INDArray getStandardDeviations(INDArray covariance) {
    INDArray sqrt = Transforms.sqrt(covariance);

    double[][] sqrtArray = sqrt.toDoubleMatrix();
    double[] diagonalArray = new double[sqrt.rows()];

    for (int i = 0; i < sqrtArray.length; i++) {
      diagonalArray[i] = sqrtArray[i][i];
    }

    return Nd4j.create(diagonalArray);
  }

  public static INDArray getKernN(INDArray standardDeviations,
                                  double scottsFactor) {
    return Transforms.round(standardDeviations.mul(scottsFactor * 2 * BigDecimalMath.PI.doubleValue()));
  }

  public static INDArray getBandwidth(INDArray covariance,
                                      double scottsFactor) {
    return InvertMatrix.invert(covariance.mul(Math.pow(scottsFactor, 2)), false);
  }

  public static Pair<INDArray, INDArray> getMeshGrid(INDArray xCoords,
                                                     INDArray yCoords) {
    int numRows = yCoords.length();
    int numCols = xCoords.length();

    xCoords = xCoords.reshape(1, numCols);
    yCoords = yCoords.reshape(numRows, 1);

    INDArray X = xCoords.repeat(0, numRows);
    INDArray Y = yCoords.repeat(1, numCols);

    return new Pair<>(X, Y);
  }

  public static INDArray getKernel(double kernNx,
                                   double kernNy,
                                   INDArray X,
                                   INDArray Y,
                                   INDArray bandwidth) {
    INDArray XFlattened = Nd4j.toFlattened(X);
    INDArray YFlattened = Nd4j.toFlattened(Y);

    INDArray kernel = Nd4j.vstack(XFlattened,
            YFlattened);
    kernel = bandwidth.mmul(kernel).mul(kernel);
    kernel = Nd4j.sum(kernel, 0).div(2.0);
    kernel = Transforms.exp(kernel.mul(-1.0));
    kernel = kernel.reshape((int) kernNy, (int) kernNx);

    return kernel;
  }

  public static INDArray convolveGrid(INDArray grid,
                                      INDArray kernel) {
    int nIn = 1;
    int nOut = 1;
    int kH = kernel.rows();
    int kW = kernel.columns();

    int mb = 1;
    int imgH = grid.rows();
    int imgW = grid.columns();

    SameDiff sd = SameDiff.create();

    INDArray wArr = kernel.reshape(nOut, nIn, kH, kW);

    INDArray inArr = grid.reshape(mb, nIn, imgH, imgW);

    SDVariable in = sd.var("in", inArr);
    SDVariable w = sd.var("W", wArr);

    SDVariable[] vars = new SDVariable[]{in, w};

    Conv2DConfig c = Conv2DConfig.builder()
            .kh(kH).kw(kW) //kernel attributes
            .ph(0).pw(0)   //padding
            .sy(1).sx(1)   //stride
            .dh(1).dw(1)   //dilation
            .isSameMode(true)
            .build();
    sd.conv2d(vars, c);
    return sd.execAndEndResult();
  }

  public static INDArray getNormalizationFactor(INDArray covariance,
                                                double scottsFactor,
                                                int n,
                                                double deltaX,
                                                double deltaY) {
    INDArray normalizationFactor = covariance.mul(2 * BigDecimalMath.PI.doubleValue()).mul(Math.pow(scottsFactor, 2));

    //TODO : for some reason this returns the correct value but not negative
    INDArray det = Nd4j.create(new double[] {0.});
    CustomOp determinant = DynamicCustomOp.builder("matrix_determinant")
            .addInputs(normalizationFactor)
            .addOutputs(det)
            .build();
    Nd4j.getExecutioner().exec(determinant);
    normalizationFactor = det;

    normalizationFactor = Transforms.sqrt(normalizationFactor).mul(n * deltaX * deltaY);

    return normalizationFactor;
  }
}
