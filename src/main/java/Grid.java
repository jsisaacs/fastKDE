import com.google.common.math.DoubleMath;

public class Grid {
  private double gridDimension;
  private Data data;

  /**
   * If no grid dimensions are given, make the default grid based off a Data object.
   * @param data : Data object that is used to calculate the Grid's dimension.
   */
  public Grid(Data data) {
    this.data = data;
    gridDimension = calculateGridDimension(data.getX().length(), true);
  }

  /**
   * Optimize the Grid's dimension if an desired grid dimension is given.
   * @param data : Data object that serves as reference for the Grid.
   * @param gridDimension : Desired grid dimension to be optimized.
   */
  public Grid(Data data, double gridDimension) {
    this.data = data;
    this.gridDimension = calculateGridDimension(gridDimension, false);
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
   * Creates a sparse 2D-Histogram of a Grid.
   * @return : Histogram
   */
  public Histogram createHistogram() {
    //TODO
    return null;
  }
}
