import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

public class Grid {
  private int gridSizeX;
  private int getGridSizeY;
  private INDArray grid;

  public Grid(Array array) {
    //TODO
  }

  public Grid(int gridSizeX, int getGridSizeY, Array array) {
    //TODO

    this.gridSizeX = gridSizeX;
    this.getGridSizeY = getGridSizeY;
  }
}
