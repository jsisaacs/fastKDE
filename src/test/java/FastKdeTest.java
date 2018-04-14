import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.assertEquals;

public class FastKdeTest {
  @Test
  public void exceptionsTest() {

  }

  @Test
  public void inputXYWeightsDimenstionTest() {
    INDArray x = Nd4j.create(new double[] {0.3395,    0.9783,    0.3895,    0.7022,    0.0185,    0.1849,    0.6926,    0.8179,    0.3845,    0.5245});
    System.out.println(x);
    INDArray y = Nd4j.create(new double[] {0.7747,    0.7053,    0.5655,    0.6056,    0.1807,    0.0696,    0.1127,    0.4404,    0.6015,    0.6682});
    System.out.println(y);
    INDArray grid = FastKde.fastKde2d(x, y);

  }
}
