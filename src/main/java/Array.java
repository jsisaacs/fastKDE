import org.nd4j.linalg.api.ndarray.INDArray;

public class Array {
  private INDArray x;
  private INDArray y;
  private INDArray weights;

  public Array(INDArray x, INDArray y) {
    this.x = x;
    this.y = y;
    arraySetup();
  }

  public Array(INDArray x, INDArray y, INDArray weights) {
    this.x = x;
    this.y = y;
    this.weights = weights;
    arraySetupWeights();
  }

  public INDArray getX() {
    return x;
  }

  public INDArray getWeights() {
    return weights;
  }

  public INDArray getY() {
    return y;
  }

  private void arraySetup() {
    //TODO
  }

  private void arraySetupWeights() {
    //TODO
  }
}
