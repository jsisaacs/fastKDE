import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class FastKdeTest {
  @Test
  public void basicArrayTest() throws Exception {
    INDArray x = Nd4j.create(1, 2);
    INDArray y = Nd4j.create(1, 2);
    Array array = new Array(x, y);
    Grid grid = new Grid(array);
  }

  @Test
  public void complexArrayTest() {

  }
}
