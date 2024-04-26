import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

def refine_dataset(data_frame):
    # Columns normalized to double and stripped of extra quotes-sj796
    return data_frame.select(*(col(column_name).cast("double").alias(column_name.strip('\"')) for column_name in data_frame.columns))

if __name__ == "__main__":
    print("Spark Session for Wine Quality Analysis was initiated by sj796")

    # Spark session creation-sj796
    spark = SparkSession.builder.appName("PredictQualityOfWine").getOrCreate()
    spark.sparkContext.setLogLevel('ERROR')

    # Data paths-sj796
    path_to_training_data = "s3://vatsanbucket/TrainingDataset.csv"
    path_to_validation_data = "s3://vatsanbucket/ValidationDataset.csv"
    path_to_output = "s3://vatsanbucket/finalmodel"

    # Training data-sj796
    print(f"Training data from: {path_to_training_data} was loaded by sj796")
    training_data = spark.read.format("csv").options(header='true', inferschema='true', sep=";").load(path_to_training_data)

    # Validation data-sj796
    print(f"Validation data from: {path_to_validation_data} was loaded by sj796")
    validation_data = spark.read.format("csv").options(header='true', inferschema='true', sep=";").load(path_to_validation_data)

    # Data preparation-sj796
    prepared_training = refine_dataset(training_data)
    prepared_validation = refine_dataset(validation_data)

    # Features for the model-sj796
    features_columns = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
                        'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
                        'pH', 'sulphates', 'alcohol', 'quality']

    # Pipeline components-sj796
    features_assembler = VectorAssembler(inputCols=features_columns, outputCol='features')
    quality_indexer = StringIndexer(inputCol="quality", outputCol="target")

    # Data caching strategy-sj796 
    prepared_training.cache()
    prepared_validation.cache()

    # RandomForest classifier-sj796
    wine_quality_classifier = RandomForestClassifier(featuresCol='features', labelCol='target',
                                                     numTrees=150, maxDepth=15, seed=150, impurity='gini')

    # The machine learning pipeline-sj796
    model_pipeline = Pipeline(stages=[quality_indexer, features_assembler, wine_quality_classifier])
    model_trained = model_pipeline.fit(prepared_training)

    # Predictions on validation data-sj796
    predictions = model_trained.transform(prepared_validation)

    # Model evaluation-sj796
    accuracy_evaluator = MulticlassClassificationEvaluator(labelCol='target', predictionCol='prediction', metricName='accuracy')
    f1_evaluator = MulticlassClassificationEvaluator(labelCol='target', predictionCol='prediction', metricName='f1')
    model_accuracy = accuracy_evaluator.evaluate(predictions)
    model_f1_score = f1_evaluator.evaluate(predictions)
    print(f"Validation Accuracy computed by sj796: {model_accuracy}")
    print(f"Validation F1 Score computed by sj796: {model_f1_score}")

    # CrossValidator setup and tuning-sj796
    tuning_parameters = ParamGridBuilder() \
        .addGrid(wine_quality_classifier.impurity, ["entropy", "gini"]) \
        .addGrid(wine_quality_classifier.numTrees, [50, 150]) \
        .addGrid(wine_quality_classifier.maxDepth, [6, 9]) \
        .addGrid(wine_quality_classifier.minInstancesPerNode, [6]) \
        .addGrid(wine_quality_classifier.seed, [100, 200]) \
        .build()
    
    cross_validation = CrossValidator(estimator=model_pipeline,
                                      estimatorParamMaps=tuning_parameters,
                                      evaluator=f1_evaluator,  # Using F1 evaluator for CrossValidator
                                      numFolds=2)
    optimal_model = cross_validation.fit(prepared_training).bestModel

    # Final evaluation-sj796
    final_predictions = optimal_model.transform(prepared_validation)
    final_accuracy = accuracy_evaluator.evaluate(final_predictions)
    final_f1_score = f1_evaluator.evaluate(final_predictions)
    print(f"Final Accuracy after Cross-Validation achieved by sj796: {final_accuracy}")
    print(f"Final F1 Score after Cross-Validation achieved by sj796: {final_f1_score}")

    print("Saving the best model to S3 by sj796")
    model_path = path_to_output
    model_trained.write().overwrite().save(model_path)

    sys.exit(0)
