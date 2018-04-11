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
    INDArray x = Nd4j.create(new double[] {0., 2., 3., 5., 5., 6., 7., 6., 5., 5., 3., 2., 1.});
    INDArray y = Nd4j.create(new double[] {1., 2.9, 3.1, 5.2, 5.3, 6.1, 7.2, 6.1, 5.1, 5.1, 3.1, 2., 1.1});
    System.out.println(FastKde.fastKde2d(x, y));
  }


}
