import org.javatuples.Pair;
import org.javatuples.Quartet;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

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

    //---------------------------------------- 2d histogram --------------------------------------------------------

    Quartet<Double, Double, Double, Double> extents = new Quartet<>(
            x.minNumber().doubleValue(),
            x.maxNumber().doubleValue(),
            y.minNumber().doubleValue(),
            y.maxNumber().doubleValue());

    double deltaX = (extents.getValue1() - extents.getValue0()) / (gridSize - 1);
    double deltaY = (extents.getValue3() - extents.getValue2()) / (gridSize - 1);

    INDArray bins = Utilities.getBins(x, y, deltaX, deltaY, extents);
    INDArray grid = Utilities.getCooMatrix(weights, bins, gridSize);

    //------------------------------------------ Kernel preliminary calculations -----------------------------------

    INDArray covariance = Utilities.getCovariance(bins, noCorrelation);
    covariance = covariance.div(0.1);
    double scottsFactor = Utilities.getScottsFactor(x.length(), adjust);

    //------------------------------------------ Make the Gaussian kernel ------------------------------------------

    INDArray standardDeviations = Utilities.getStandardDeviations(covariance);
    INDArray kernN = Utilities.getKernN(standardDeviations, scottsFactor);

    double kernNx = kernN.getDouble(0, 0);
    double kernNy = kernN.getDouble(0, 1);

    INDArray bandwidth = Utilities.getBandwidth(covariance, scottsFactor);

    INDArray xCoords = Nd4j.arange(kernNx).sub(kernNx / 2.0);
    INDArray yCoords = Nd4j.arange(kernNy).sub(kernNy / 2.0);
    Pair<INDArray, INDArray> meshGrid = Utilities.getMeshGrid(xCoords, yCoords);

    INDArray kernel = Utilities.getKernel(kernNx, kernNy,
            meshGrid.getValue0(), meshGrid.getValue1(), bandwidth);

    //------------------------------------------ Produce the kernel density estimate -------------------------------

    grid = Utilities.convolveGrid(grid, kernel);

    INDArray normalizationFactor = Utilities.getNormalizationFactor(covariance,
            scottsFactor, x.length(), deltaX, deltaY);

    grid = grid.div(normalizationFactor);
    grid = grid.reshape(gridSize, gridSize);
    Nd4j.rot90(grid);

    return grid;
  }

  public static INDArray fastKde2d(INDArray x, INDArray y) {
    x = Nd4j.stripOnes(x);
    y = Nd4j.stripOnes(y);
    INDArray weights = Nd4j.create(x.length()).add(1);

    return fastKde2d(x, y, 200, false, weights, 1.0);
  }
}

