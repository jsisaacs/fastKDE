import com.google.common.math.DoubleMath;
import org.javatuples.Quartet;

public class Grid {
  private double gridDimension;
  private Data data;
  private Quartet<Double, Double, Double, Double> extents;

  /**
   * If no grid dimensions are given, make the default grid based off a Data object.
   * Constructs the extents of the Data.
   * @param data : Data object that is used to calculate the Grid's dimension.
   */
  public Grid(Data data) {
    this.data = data;
    gridDimension = calculateGridDimension(data.getX().length(), true);
    extents = new Quartet<>(
            data.getX().minNumber().doubleValue(),
            data.getX().maxNumber().doubleValue(),
            data.getY().minNumber().doubleValue(),
            data.getY().maxNumber().doubleValue()
    );
  }

  /**
   * Optimize the Grid's dimension if an desired grid dimension is given.
   * Constructs the extents of the Data.
   * @param data : Data object that serves as reference for the Grid.
   * @param gridDimension : Desired grid dimension to be optimized.
   */
  public Grid(Data data, double gridDimension) {
    this.data = data;
    this.gridDimension = calculateGridDimension(gridDimension, false);
    extents = new Quartet<>(
            data.getX().minNumber().doubleValue(),
            data.getX().maxNumber().doubleValue(),
            data.getY().minNumber().doubleValue(),
            data.getY().maxNumber().doubleValue()
    );
  }

  /**
   *  Calculates the optimal grid dimension.
   * @param gridDimension : Desired grid dimension to be optimized.
   * @param defaultGrid : True if the Grid uses dimensions based off Data, false if otherwise.
   * @return : Optimal grid dimension.
   */
  private static double calculateGridDimension(double gridDimension, boolean defaultGrid) {
    if (defaultGrid) {
      gridDimension = Math.max(gridDimension, 512.0);
    }
    gridDimension = Math.pow(2, Math.ceil(DoubleMath.log2(gridDimension)));
    return gridDimension;
  }

  /**
   * @return : data
   */
  public Data getData() {
    return data;
  }

  /**
   * @return : gridDimension
   */
  public double getGridDimension() {
    return gridDimension;
  }

  /**
   * @return : minimum value of X
   */
  public double getXMin() {
    return extents.getValue0();
  }

  /**
   * @return : maximum value of X
   */
  public double getXMax() {
    return extents.getValue1();
  }

  /**
   * @return : minimum value of Y
   */
  public double getYMin() {
    return extents.getValue2();
  }

  /**
   * @return : maximum value of Y
   */
  public double getYMax() {
    return extents.getValue3();
  }
}
