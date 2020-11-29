package org.apache.spark.ml.made

import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.{DataFrame}
import org.apache.spark.sql.SparkSession
import breeze.linalg.{*, DenseMatrix, DenseVector}
import breeze.numerics._
import breeze.stats.mean
import org.scalatest._
import flatspec._
import matchers._

trait WithSpark {
  val spark = WithSpark._spark
  val sqlc = WithSpark._sqlc
}

object WithSpark {
  val _spark = SparkSession.builder
    .appName("Linear Regression")
    .master("local[4]")
    .getOrCreate()

  val _sqlc = _spark.sqlContext

  _spark.sparkContext.setLogLevel("ERROR")
}

class LinearRegressionTest extends AnyFlatSpec with should.Matchers with WithSpark {

  val data: DataFrame = LinearRegressionTest._data
  val coefficients: linalg.DenseVector = LinearRegressionTest._coefficients
  val X: DenseMatrix[Double] = LinearRegressionTest._X
  val y: DenseVector[Double] = LinearRegressionTest._y

  val delta = 0.01

  "Estimator" should "train" in {
    val estimator: LinearRegression = new LinearRegression()
      .setFeaturesCol("features")
      .setLabelCol("label")
      .setPredictionCol("prediction")

    val model = estimator.fit(data)

    model.coefficients(0) should be(coefficients(0) +- delta)
    model.coefficients(1) should be(coefficients(1) +- delta)
    model.coefficients(2) should be(coefficients(2) +- delta)
  }

  "Model" should "predict" in {
    val model: LinearRegressionModel = new LinearRegressionModel(
      coefficients
    ).setFeaturesCol("features")
      .setLabelCol("label")
      .setPredictionCol("prediction")

    val pred = DenseVector(model.transform(data).select("prediction").collect().map(x => x.getAs[Vector](0)(0)))

    sqrt(mean(pow(pred - y, 2))) should be(0.0 +- delta)
  }
}

object LinearRegressionTest extends WithSpark {
    val _X: DenseMatrix[Double] = DenseMatrix.rand[Double](100000, 3)
    val _coefficients: linalg.DenseVector = Vectors.dense(1.5, 0.3, -0.7).toDense
    val _y: DenseVector[Double] = _X * _coefficients.asBreeze

    val _data: DataFrame = {
      import sqlc.implicits._

      val tmp = DenseMatrix.horzcat(_X, _y.asDenseMatrix.t)
      val df = tmp(*, ::).iterator
        .map(x => (x(0), x(1), x(2), x(3)))
        .toSeq
        .toDF("x1", "x2", "x3", "label")

      val assembler = new VectorAssembler()
        .setInputCols(Array("x1", "x2", "x3"))
        .setOutputCol("features")
      val output = assembler.transform(df).select("features", "label")

      output
    }
  }
