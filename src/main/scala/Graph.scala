import co.theasi.plotly._
import org.nd4j.linalg.api.ndarray.INDArray

object Graph {
  def kde1d(x: INDArray): Unit = {

  }

  def kde2d(x: INDArray, y: INDArray): Unit = {

  }

  def main(args: Array[String]): Unit = {
    val xs = (0.0 to 2.0 by 0.1)
    val ys = xs.map { x => x*x }

    val plot = Plot().withScatter(xs, ys)

    draw(plot, "my-first-plot")
  }
}
