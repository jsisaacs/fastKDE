/*
Scala implementation of Morgan Fouesneau's fastKDE in the 'faststats' library,
originally written in Python.

https://github.com/mfouesneau/faststats

The MIT License (MIT)

Copyright (c) 2013 Morgan Fouesneau & contributors

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

import org.nd4j.linalg.api.ndarray.INDArray

object FastKDE {

  /**
    * A FFT-based Gaussian kernel density estimate (KDE) for computing the KDE on a regular grid
    *
    * @param x: INDArray; x-coordinates of the input data points
    * @param y: INDArray; y-coordinates of the input data points
    * @param gridSize: Tuple2; a (nx, ny) tuple of the size of the output grid (default: 200x200)
    * @param extents: Tuple4; (xmin, xmax, ymin, ymax) tuple of the extents of output grid
    *               (default: extent of input data)
    * @param noCorrelation: Boolean; if true, the correlation between the x and y coordinates
    *                     will be ignored when performing the KDE (default: false)
    * @param weights: INDArray; an array of the same shape as x & y that weights each sample
    *               (xI, yI) by each value in weights (wI). defaults to an array of ones the
    *               same size as x & y (default: None)
    * @param adjust: Float; an adjustment factor for the bandwidth, where bandwidth = bandwidth * adjust
    * @return (INDArray, (Int, Int, Int, Int)): Tuple2; a gridded 2D KDE of the input points, and a Tuple4
    *         with extents of it
    */

  def fastKDE(x: INDArray, y: INDArray, gridSize: (Int, Int) = (200, 200),
              extents: Option[(Int, Int, Int, Int)] = None, noCorrelation: Boolean = false,
              weights: Option[INDArray] = None, adjust: Float = 1): (INDArray, (Int, Int, Int, Int)) = {
    //TODO
    null
  }

  /**
    * Makes sure the x and y inputs are INDArrays and that they are the same size.
    * Weights all points equally by default and sets the weights given a custom weight
    * INTArray.
    *
    * @param x
    * @param y
    * @param weights
    */

  def arraySetup(x: INDArray, y: INDArray, weights: Option[INDArray] = None): Unit = {
    //TODO
  }

  /**
    * Makes the grid and discretizes the data and rounds it to the next power of 2
    * to optimize the FFT use.
    *
    * @param x
    * @param y
    * @param gridSize
    */

  def optimizeGridsize(x: INDArray, y: INDArray, gridSize: (Int, Int) = (200, 200)): Unit = {
    //TODO
  }

  /**
    * Calculates the covariance matrix and the scaling factor for bandwidth.
    *
    * 
    */

  def kernelPreliminaryCalculations(): Unit = {
    //TODO
  }

  /**
    * Makes the gaussian kernel, determines the bandwidth to use,
    * and evaluates the function on the kernel grid.
    */

  def gaussianKernel(): Unit = {
    //TODO
  }

  /**
    * Nomalizes the output and gives the KDE estimation.
    */

  def kdeEstimate(): Unit = {
    //TODO
  }

  def main(args: Array[String]): Unit = {

  }
}
