import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.factory.Nd4j;

public class Array {
  private INDArray x;
  private INDArray y;
  private INDArray weights;

  /**
   * Create an Array without custom weights.
   * @param x : x-coordinates of the input data points
   * @param y : y-coordinates of the input data points
   * @throws Exception : if x and y have different lengths
   */
  public Array(INDArray x, INDArray y) throws Exception {
    this.x = x;
    this.y = y;

    //Throw an exception if x and y have different lengths.
    if (this.getX().length() != this.getY().length()) {
      throw new Exception("x and y have different lengths.");
    }

    //If no custom weights are given, create a default INDArray of weights
    //with values of 1.
    int length = this.getX().length();
    weights = Nd4j.ones(length);
  }

  /**
   * Create an Array with custom weights.
   * @param x : x-coordinates of the input data points
   * @param y : y-coordinates of the input data points
   * @param weights : an array of the same shape as x & y that weights each sample
   *                by each value in weights
   * @throws Exception : if x and y have different lengths
   * @throws Exception : weights and x/y have different lengths
   */
  public Array(INDArray x, INDArray y, INDArray weights) throws Exception {
    this.x = x;
    this.y = y;
    this.weights = weights;

    //Throw an exception if x and y have different lengths.
    if (this.getX().length() != this.getY().length()) {
      throw new Exception("x and y have different lengths.");
    }

    //Reshape weights to remove single-dimensional entries.
    int[] shape = Shape.squeeze(weights.shape());
    this.weights = weights.reshape(shape);

    //Throw an exception if weights and x/y have different lengths.
    if (this.weights.length() != this.getX().length()) {
      throw new Exception("weights and x/y have different lengths.");
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
}
