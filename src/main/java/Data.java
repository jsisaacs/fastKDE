import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;

public class Data {
  private INDArray x;
  private INDArray y;
  private INDArray weights;

  /**
   * Create a Data object without custom weights.
   * @param x : X-coordinates of the input data points.
   * @param y : Y-coordinates of the input data points.
   * @throws IndexOutOfBoundsException : If x or y is out of bounds.
   * @throws IllegalArgumentException : If x and y have different lengths.
   */
  public Data(INDArray x, INDArray y) {
    this.x = x;
    this.y = y;

    //Throw an exception if x or y is out of bounds.
    if (x.length() <= 0 || y.length() <= 0) {
      throw new IndexOutOfBoundsException("x or y is out of bounds.");
    }

    //Throw an exception if x and y have different lengths.
    if (this.getX().length() != this.getY().length()) {
      throw new IllegalArgumentException("x and y have different lengths.");
    }

    //If no custom weights are given, create a default INDArray of weights
    //with values of 1.
    int length = this.getX().length();
    weights = Nd4j.ones(length);
  }

  /**
   * Create a Data object with custom weights.
   * @param x : X-coordinates of the input data points.
   * @param y : Y-coordinates of the input data points.
   * @param weights : An array of the same shape as x & y that weights each sample
   *                by each value in weights.
   * @throws IndexOutOfBoundsException : If x or y is out of bounds.
   * @throws IllegalArgumentException : If x and y have different lengths.
   */
  public Data(INDArray x, INDArray y, INDArray weights) {
    this.x = x;
    this.y = y;
    this.weights = weights;

    //Throw an exception if x or y is out of bounds.
    if (x.length() <= 0 || y.length() <= 0) {
      throw new IndexOutOfBoundsException("x or y is out of bounds.");
    }

    //Throw an exception if x and y have different lengths.
    if (this.getX().length() != this.getY().length()) {
      throw new IllegalArgumentException("x and y have different lengths.");
    }

    //Reshape weights to remove single-dimensional entries.
    int[] shape = Shape.squeeze(weights.shape());
    this.weights = weights.reshape(shape);

    //Throw an exception if weights and x/y have different lengths.
    if (this.weights.length() != this.getX().length()) {
      throw new IllegalArgumentException("weights and x/y have different lengths.");
    }
  }

  /**
   * @return : x
   */
  public INDArray getX() {
    return this.x;
  }

  /**
   * @return : y
   */
  public INDArray getY() {
    return this.y;
  }

  /**
   * @return : weights
   */
  public INDArray getWeights() {
    return this.weights;
  }

  //TODO : use Nd4j.writeTxt to convert grid to JSON to plot in python

  public static void main(String[] args) throws IOException {
    double[] testArray = {1.0, 2.0, 3.0};
    INDArray testX = Nd4j.create(testArray);
    INDArray testY = Nd4j.create(testArray);
    //Data data = new Data(testX, testY);
    File file = new File("outputTest.txt");
    Nd4j.writeTxt(testX, "output.json");
  }
}
