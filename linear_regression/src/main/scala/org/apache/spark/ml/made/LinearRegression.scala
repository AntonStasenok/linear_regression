package org.apache.spark.ml.made

import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.apache.spark.sql.{DataFrame, Dataset, Encoder}
import org.apache.spark.sql.types.{DoubleType, StructField, StructType}
import org.apache.spark.ml.linalg.{DenseVector, Vector, VectorUDT, Vectors}
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.param.shared.{HasFeaturesCol, HasLabelCol, HasMaxIter, HasPredictionCol, HasStepSize}
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable, MetadataUtils, SchemaUtils}
import org.apache.spark.mllib.stat.MultivariateOnlineSummarizer
import org.apache.spark.mllib
import breeze.linalg._

trait LinearRegressionParams extends HasLabelCol with HasFeaturesCol with HasPredictionCol
  with HasMaxIter with HasStepSize {

  def setLabelCol(value: String): this.type = set(labelCol, value)
  def setFeaturesCol(value: String): this.type = set(featuresCol, value)
  def setPredictionCol(value: String): this.type = set(predictionCol, value)
  protected def validateAndTransformSchema(schema: StructType): StructType = {
    SchemaUtils.checkColumnType(schema, getFeaturesCol, new VectorUDT())
    if (schema.fieldNames.contains($(predictionCol))) {
      SchemaUtils.checkColumnType(schema, getPredictionCol, new VectorUDT())
    } else {
      SchemaUtils.appendColumn(schema, StructField(getPredictionCol, new VectorUDT()))
    }
    if (schema.fieldNames.contains($(labelCol))) {
      SchemaUtils.checkColumnType(schema, getLabelCol, DoubleType)
    } else {
      SchemaUtils.appendColumn(schema, StructField(getLabelCol, DoubleType))
    }
    schema
  }
}

class LinearRegression(override val uid: String) extends Estimator[LinearRegressionModel] with LinearRegressionParams
  with DefaultParamsWritable {
  def this() = this(Identifiable.randomUID("linearRegression"))
  def setMaxIter(value: Int): this.type = set(maxIter, value)
  setDefault(maxIter, 150)
  def setStepSize(value: Double): this.type = set(stepSize, value)
  setDefault(stepSize, 0.5)
  override def fit(dataset: Dataset[_]): LinearRegressionModel = {
    implicit val encoder: Encoder[Vector] = ExpressionEncoder()
    val assembler = new VectorAssembler()
      .setInputCols(Array($(featuresCol), $(labelCol)))
      .setOutputCol("feats")
    val features = assembler
      .transform(dataset)
      .select("feats").as[Vector]
    val n_features = MetadataUtils.getNumFeatures(dataset, $(featuresCol))
    var coefficients = breeze.linalg.DenseVector.rand[Double](n_features )
    for (_ <- 0 to $(maxIter)) {
      val summary = features.rdd.mapPartitions((data: Iterator[Vector]) => {
        val summarizer = new MultivariateOnlineSummarizer()
        data.foreach(v => {
          val X = v.asBreeze(0 until coefficients.size).toDenseVector
          val y = v.asBreeze(-1)
          val grads = X * (breeze.linalg.sum(X * coefficients) - y)
          summarizer.add(mllib.linalg.Vectors.fromBreeze(grads))
        })
        Iterator(summarizer)
      }).reduce(_ merge _)
      coefficients = coefficients - $(stepSize) * summary.mean.asBreeze
    }
    copyValues(new LinearRegressionModel(
      Vectors.fromBreeze(coefficients(0 until coefficients.size)).toDense)
    ).setParent(this)
  }
  override def copy(extra: ParamMap): Estimator[LinearRegressionModel] = defaultCopy(extra)
  override def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)
}

object LinearRegression extends DefaultParamsReadable[LinearRegression]

class LinearRegressionModel private[made](
                                           override val uid: String,
                                           val coefficients: DenseVector) extends Model[LinearRegressionModel]
  with LinearRegressionParams {
  private[made] def this(coefficients: DenseVector) =
    this(Identifiable.randomUID("linearRegressionModel"), coefficients)
  override def copy(extra: ParamMap): LinearRegressionModel = copyValues(
    new LinearRegressionModel(uid, coefficients/*, intercept*/))
  override def transform(dataset: Dataset[_]): DataFrame = {
    val transformUdf = dataset.sqlContext.udf.register(uid + "_predict",
      (x: Vector) => {
        Vectors.fromBreeze(breeze.linalg.DenseVector(coefficients.asBreeze.dot(x.asBreeze)))
      })
    dataset.withColumn($(predictionCol), transformUdf(dataset($(featuresCol))))
  }
  override def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)
}
