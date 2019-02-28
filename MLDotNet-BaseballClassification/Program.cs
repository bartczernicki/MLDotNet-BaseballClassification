using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Google.Protobuf;
using Microsoft.Data.DataView; // Required for Dataview
using Microsoft.ML;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Normalizers;

using System.Diagnostics;

namespace MLDotNet_BaseballClassification
{
    class Program
    {
        private static string _appPath => Path.GetDirectoryName(Environment.GetCommandLineArgs()[0]);
        private static string _trainDataPath => Path.Combine(_appPath, "..", "..", "..", "Data", "BaseballHOFTrainingv2.csv");
        private static string _validationDataPath => Path.Combine(_appPath, "..", "..", "..", "Data", "BaseballHOFValidationv2.csv");

        private static MLContext _mlContext;

        private static string _labelColunmn = "OnHallOfFameBallot";

        private static string _modelPath => Path.Combine(_appPath, "..", "..", "..", "Models", string.Format("model-{0}.zip", _labelColunmn));
        private static string _OnnxModelPath => Path.Combine(_appPath, "..", "..", "..", "Models", string.Format("model-{0}.onnx", _labelColunmn));

        // List of columns used for training
        // Comment out (or uncomment) feature names in order to explicitly select features for model training
        private static string[] featureColumns = new string[] {
            //"YearsPlayed", "AB", "R", "H", "Doubles", "Triples", "HR", "RBI", "SB",
            "BattingAverage", "SluggingPct", "AllStarAppearances", "MVPs", "TripleCrowns", "GoldGloves",
            "MajorLeaguePlayerOfTheYearAwards", "TB"
        };

        static void Main(string[] args)
        {
            Stopwatch sw = new Stopwatch();
            sw.Start();

            Console.WriteLine("Starting Baseball Predictions - Model Job");
            Console.WriteLine("This job will build a series of models that will predict both:");
            Console.WriteLine("Whether a player would make it on the HOF Ballot & would be inducted to the HOF.\n");

            #region Step 1) ML.NET Setup & Load Data

            Console.WriteLine("##########################");
            Console.WriteLine("Step 1: Load Data...");
            Console.WriteLine("##########################\n");

            // Set the seed for reproducability
            _mlContext = new MLContext(seed: 100);

            // Read the data from a text file
            var dataTrain = _mlContext.Data.ReadFromTextFile<MLBBaseballBatter>(path: _trainDataPath,
                hasHeader: true, separatorChar: ',', allowQuotedStrings: false);
            var dataValidation = _mlContext.Data.ReadFromTextFile<MLBBaseballBatter>(path: _validationDataPath,
                hasHeader: true, separatorChar: ',', allowQuotedStrings: false);

            // Debug Only: Preview the data
            var dataTrainPreview = dataTrain.Preview();
            var dataValidationPreview = dataValidation.Preview();

            // Cache the loaded data
            var cachedTrainData = _mlContext.Data.Cache(dataTrain);
            var cachedValidationData = _mlContext.Data.Cache(dataValidation);

            #endregion

            #region Step 2) Build Multiple Machine Learning Models

            Console.WriteLine("##########################");
            Console.WriteLine("Step 2: Train Models...");
            Console.WriteLine("##########################\n");


            /* LIGHTGBM MODELS */
            Console.WriteLine("Training...LightGbm Models.");

            _labelColunmn = "OnHallOfFameBallot";
            // Build simple data pipeline
            var learningPipelineLightGbmOnHallOfFameBallot =
                GetBaseLinePipeline().Append(
                _mlContext.BinaryClassification.Trainers.LightGbm(labelColumn: _labelColunmn)
                );
            // Fit (build a Machine Learning Model)
            var modelLightGbmOnHallOfFameBallot = learningPipelineLightGbmOnHallOfFameBallot.Fit(cachedTrainData);
            // Save the model to storage
            SaveModel("LightGbm", _labelColunmn, modelLightGbmOnHallOfFameBallot);
            SaveOnnxModel("LightGbm", _labelColunmn, modelLightGbmOnHallOfFameBallot, _mlContext, cachedTrainData);

            _labelColunmn = "InductedToHallOfFame";
            // Build simple data pipeline
            var learningPipelineLightGbmInductedToHallOfFame =
                GetBaseLinePipeline().Append(
                _mlContext.BinaryClassification.Trainers.LightGbm(labelColumn: _labelColunmn)
                );
            // Fit (build a Machine Learning Model)
            var modelLightGbmInductedToHallOfFame = learningPipelineLightGbmInductedToHallOfFame.Fit(cachedTrainData);
            // Save the model to storage
            SaveModel("LightGbm", _labelColunmn, modelLightGbmInductedToHallOfFame);
            SaveOnnxModel("LightGbm", _labelColunmn, modelLightGbmInductedToHallOfFame, _mlContext, cachedTrainData);


            /* LOGISTIC REGRESSION MODELS */
            Console.WriteLine("Training...Logistic Regression Models.");

            _labelColunmn = "OnHallOfFameBallot";
            // Build simple data pipeline
            var learningPipelineLogisticRegressionOnHallOfFameBallot =
                GetBaseLinePipeline().Append(
                _mlContext.BinaryClassification.Trainers.LogisticRegression(labelColumn: _labelColunmn, l1Weight: 0.4f, l2Weight: 0.4f)
                );
            // Fit (build a Machine Learning Model)
            var modelLogisticRegressionOnHallOfFameBallot = learningPipelineLogisticRegressionOnHallOfFameBallot.Fit(cachedTrainData);
            // Save the model to storage
            SaveModel("LogisticRegression", _labelColunmn, modelLogisticRegressionOnHallOfFameBallot);
            SaveOnnxModel("LogisticRegression", _labelColunmn, modelLogisticRegressionOnHallOfFameBallot, _mlContext, cachedTrainData);

            _labelColunmn = "InductedToHallOfFame";
            // Build simple data pipeline
            var learningPipelineLogisticRegressionInductedToHallOfFame =
                GetBaseLinePipeline().Append(
                _mlContext.BinaryClassification.Trainers.LogisticRegression(labelColumn: _labelColunmn, l1Weight: 0.4f, l2Weight: 0.4f)
                );
            // Fit (build a Machine Learning Model)
            var modelLogisticRegressionInductedToHallOfFame = learningPipelineLogisticRegressionInductedToHallOfFame.Fit(cachedTrainData);
            // Save the model to storage
            SaveModel("LogisticRegression", _labelColunmn, modelLogisticRegressionInductedToHallOfFame);
            SaveOnnxModel("LogisticRegression", _labelColunmn, modelLogisticRegressionInductedToHallOfFame, _mlContext, cachedTrainData);


            /* AVERAGED PERCEPTRON MODELS */
            Console.WriteLine("Training...Averaged Perceptron Models.");

            _labelColunmn = "OnHallOfFameBallot";
            // Build simple data pipeline
            var learningPipelineAveragedPerceptronOnHallOfFameBallot =
                GetBaseLinePipeline().Append(
                _mlContext.BinaryClassification.Trainers.AveragedPerceptron(labelColumn: _labelColunmn)
                );
            // Fit (build a Machine Learning Model)
            var modelAveragedPerceptronOnHallOfFameBallot = learningPipelineAveragedPerceptronOnHallOfFameBallot.Fit(cachedTrainData);
            // Save the model to storage
            SaveModel("AveragedPerceptron", _labelColunmn, modelAveragedPerceptronOnHallOfFameBallot);
            SaveOnnxModel("AveragedPerceptron", _labelColunmn, modelAveragedPerceptronOnHallOfFameBallot, _mlContext, cachedTrainData);

            _labelColunmn = "InductedToHallOfFame";
            // Build simple data pipeline
            var learningPipelineAveragedPerceptronInductedToHallOfFame =
                GetBaseLinePipeline().Append(
                _mlContext.BinaryClassification.Trainers.AveragedPerceptron(labelColumn: _labelColunmn)
                );
            // Fit (build a Machine Learning Model)
            var modelAveragedPerceptronInductedToHallOfFame = learningPipelineAveragedPerceptronInductedToHallOfFame.Fit(cachedTrainData);
            // Save the model to storage
            SaveModel("AveragedPerceptron", _labelColunmn, modelAveragedPerceptronInductedToHallOfFame);
            SaveOnnxModel("AveragedPerceptron", _labelColunmn, modelAveragedPerceptronInductedToHallOfFame, _mlContext, cachedTrainData);


            /* FAST FOREST MODELS */
            Console.WriteLine("Training...Fast Forest Models.");

            _labelColunmn = "OnHallOfFameBallot";
            // Build simple data pipeline
            var learningPipelineFastForestOnHallOfFameBallot =
                GetBaseLinePipeline().Append(
                _mlContext.BinaryClassification.Trainers.FastForest(labelColumn: _labelColunmn)
                );
            // Fit (build a Machine Learning Model)
            var modelFastForestOnHallOfFameBallot = learningPipelineFastForestOnHallOfFameBallot.Fit(cachedTrainData);
            // Save the model to storage
            SaveModel("FastForest", _labelColunmn, modelFastForestOnHallOfFameBallot);
            SaveOnnxModel("FastForest", _labelColunmn, modelFastForestOnHallOfFameBallot, _mlContext, cachedTrainData);

            _labelColunmn = "InductedToHallOfFame";
            // Build simple data pipeline
            var learningPipelineFastForestInductedToHallOfFame =
                GetBaseLinePipeline().Append(
                _mlContext.BinaryClassification.Trainers.FastForest(labelColumn: _labelColunmn)
                );
            // Fit (build a Machine Learning Model)
            var modelFastForestInductedToHallOfFame = learningPipelineFastForestInductedToHallOfFame.Fit(cachedTrainData);
            // Save the model to storage
            SaveModel("FastForest", _labelColunmn, modelFastForestInductedToHallOfFame);
            SaveOnnxModel("FastForest", _labelColunmn, modelFastForestInductedToHallOfFame, _mlContext, cachedTrainData);


            /* FAST TREE MODELS */
            Console.WriteLine("Training...Fast Tree Models.");

            _labelColunmn = "OnHallOfFameBallot";
            // Build simple data pipeline
            var learningPipelineFastTreeOnHallOfFameBallot =
                GetBaseLinePipeline().Append(
                _mlContext.BinaryClassification.Trainers.FastTree(labelColumn: _labelColunmn)
                );
            // Fit (build a Machine Learning Model)
            var modelFastTreeOnHallOfFameBallot = learningPipelineFastTreeOnHallOfFameBallot.Fit(cachedTrainData);
            // Save the model to storage
            SaveModel("FastTree", _labelColunmn, modelFastTreeOnHallOfFameBallot);
            SaveOnnxModel("FastTree", _labelColunmn, modelFastTreeOnHallOfFameBallot, _mlContext, cachedTrainData);

            _labelColunmn = "InductedToHallOfFame";
            // Build simple data pipeline
            var learningPipelineFastTreeInductedToHallOfFame =
                GetBaseLinePipeline().Append(
                _mlContext.BinaryClassification.Trainers.FastTree(labelColumn: _labelColunmn)
                );
            // Fit (build a Machine Learning Model)
            var modelFastTreeInductedToHallOfFame = learningPipelineFastTreeInductedToHallOfFame.Fit(cachedTrainData);
            // Save the model to storage
            SaveModel("FastTree", _labelColunmn, modelFastTreeInductedToHallOfFame);
            SaveOnnxModel("FastTree", _labelColunmn, modelFastTreeInductedToHallOfFame, _mlContext, cachedTrainData);


            /* STOCHASTIC GRADIENT DESCENT MODELS */
            Console.WriteLine("Training...Stochastic Gradient Descent Models.");

            _labelColunmn = "OnHallOfFameBallot";
            // Build simple data pipeline
            var learningPipelineStochasticGradientDescentOnHallOfFameBallot =
                GetBaseLinePipeline().Append(
                _mlContext.BinaryClassification.Trainers.StochasticGradientDescent(labelColumn: _labelColunmn)
                );
            // Fit (build a Machine Learning Model)
            var modelStochasticGradientDescentOnHallOfFameBallot = learningPipelineStochasticGradientDescentOnHallOfFameBallot.Fit(cachedTrainData);
            // Save the model to storage
            SaveModel("StochasticGradientDescent", _labelColunmn, modelStochasticGradientDescentOnHallOfFameBallot);
            SaveOnnxModel("StochasticGradientDescent", _labelColunmn, modelStochasticGradientDescentOnHallOfFameBallot, _mlContext, cachedTrainData);

            _labelColunmn = "InductedToHallOfFame";
            // Build simple data pipeline
            var learningPipelineStochasticGradientDescentInductedToHallOfFame =
                GetBaseLinePipeline().Append(
                _mlContext.BinaryClassification.Trainers.StochasticGradientDescent(labelColumn: _labelColunmn)
                );
            // Fit (build a Machine Learning Model)
            var modelStochasticGradientDescentInductedToHallOfFame = learningPipelineStochasticGradientDescentInductedToHallOfFame.Fit(cachedTrainData);
            // Save the model to storage
            SaveModel("StochasticGradientDescent", _labelColunmn, modelStochasticGradientDescentInductedToHallOfFame);
            SaveOnnxModel("StochasticGradientDescent", _labelColunmn, modelStochasticGradientDescentInductedToHallOfFame, _mlContext, cachedTrainData);


            /* STOCHASTIC DUAL COORDINATE ASCENT MODELS */
            Console.WriteLine("Training...Stochastic Dual Coordinate Ascent Models.");

            _labelColunmn = "OnHallOfFameBallot";
            // Build simple data pipeline
            var learningPipelineStochasticDualCoordinateAscentOnHallOfFameBallot =
                GetBaseLinePipeline().Append(
                _mlContext.BinaryClassification.Trainers.StochasticDualCoordinateAscent(labelColumn: _labelColunmn)
                );
            // Fit (build a Machine Learning Model)
            var modelStochasticDualCoordinateAscentOnHallOfFameBallot = learningPipelineStochasticDualCoordinateAscentOnHallOfFameBallot.Fit(cachedTrainData);
            // Save the model to storage
            SaveModel("StochasticDualCoordinateAscent", _labelColunmn, modelStochasticDualCoordinateAscentOnHallOfFameBallot);
            SaveOnnxModel("StochasticDualCoordinateAscent", _labelColunmn, modelStochasticDualCoordinateAscentOnHallOfFameBallot, _mlContext, cachedTrainData);

            _labelColunmn = "InductedToHallOfFame";
            // Build simple data pipeline
            var learningPipelineStochasticDualCoordinateAscentInductedToHallOfFame =
                GetBaseLinePipeline().Append(
                _mlContext.BinaryClassification.Trainers.StochasticDualCoordinateAscent(labelColumn: _labelColunmn)
                );
            // Fit (build a Machine Learning Model)
            var modelStochasticDualCoordinateAscentInductedToHallOfFame = learningPipelineStochasticDualCoordinateAscentInductedToHallOfFame.Fit(cachedTrainData);
            // Save the model to storage
            SaveModel("StochasticDualCoordinateAscent", _labelColunmn, modelStochasticDualCoordinateAscentInductedToHallOfFame);
            SaveOnnxModel("StochasticDualCoordinateAscent", _labelColunmn, modelStochasticDualCoordinateAscentInductedToHallOfFame, _mlContext, cachedTrainData);


            /* GENERALIZED ADDITIVE MODELS */
            Console.WriteLine("Training...Generalized Additive Models.");

            _labelColunmn = "OnHallOfFameBallot";
            // Build simple data pipeline
            var learningPipelineGeneralizedAdditiveModelsOnHallOfFameBallot =
                GetBaseLinePipeline().Append(
                _mlContext.BinaryClassification.Trainers.GeneralizedAdditiveModels(labelColumn: _labelColunmn)
                );
            // Fit (build a Machine Learning Model)
            var modelGeneralizedAdditiveModelsOnHallOfFameBallot = learningPipelineGeneralizedAdditiveModelsOnHallOfFameBallot.Fit(cachedTrainData);
            // Save the model to storage
            SaveModel("GeneralizedAdditiveModels", _labelColunmn, modelGeneralizedAdditiveModelsOnHallOfFameBallot);
            SaveOnnxModel("GeneralizedAdditiveModels", _labelColunmn, modelGeneralizedAdditiveModelsOnHallOfFameBallot, _mlContext, cachedTrainData);

            _labelColunmn = "InductedToHallOfFame";
            // Build simple data pipeline
            var learningPipelineGeneralizedAdditiveModelsInductedToHallOfFame =
                GetBaseLinePipeline().Append(
                _mlContext.BinaryClassification.Trainers.GeneralizedAdditiveModels(labelColumn: _labelColunmn)
                );
            // Fit (build a Machine Learning Model)
            var modelGeneralizedAdditiveModelsInductedToHallOfFame = learningPipelineGeneralizedAdditiveModelsInductedToHallOfFame.Fit(cachedTrainData);
            // Save the model to storage
            SaveModel("GeneralizedAdditiveModels", _labelColunmn, modelGeneralizedAdditiveModelsInductedToHallOfFame);
            SaveOnnxModel("GeneralizedAdditiveModels", _labelColunmn, modelGeneralizedAdditiveModelsInductedToHallOfFame, _mlContext, cachedTrainData);


            /* LINEAR SUPPORT VECTOR MODELS */
            Console.WriteLine("Training...Linear Support Vector Models.");

            _labelColunmn = "OnHallOfFameBallot";
            // Build simple data pipeline
            var learningPipelineLinearSupportVectorMachinesOnHallOfFameBallot =
                GetBaseLinePipeline().Append(
                _mlContext.BinaryClassification.Trainers.LinearSupportVectorMachines(labelColumn: _labelColunmn)
                );
            // Fit (build a Machine Learning Model)
            var modelLinearSupportVectorMachinesOnHallOfFameBallot = learningPipelineLinearSupportVectorMachinesOnHallOfFameBallot.Fit(cachedTrainData);
            // Save the model to storage
            SaveModel("LinearSupportVectorMachines", _labelColunmn, modelLinearSupportVectorMachinesOnHallOfFameBallot);
            SaveOnnxModel("LinearSupportVectorMachines", _labelColunmn, modelLinearSupportVectorMachinesOnHallOfFameBallot, _mlContext, cachedTrainData);

            _labelColunmn = "InductedToHallOfFame";
            // Build simple data pipeline
            var learningPipelineLinearSupportVectorMachinesInductedToHallOfFame =
                GetBaseLinePipeline().Append(
                _mlContext.BinaryClassification.Trainers.LinearSupportVectorMachines(labelColumn: _labelColunmn)
                );
            // Fit (build a Machine Learning Model)
            var modelLinearSupportVectorMachinesInductedToHallOfFame = learningPipelineLinearSupportVectorMachinesInductedToHallOfFame.Fit(cachedTrainData);
            // Save the model to storage
            SaveModel("LinearSupportVectorMachines", _labelColunmn, modelLinearSupportVectorMachinesInductedToHallOfFame);
            SaveOnnxModel("LinearSupportVectorMachines", _labelColunmn, modelLinearSupportVectorMachinesInductedToHallOfFame, _mlContext, cachedTrainData);


            //var test = _mlContext.BinaryClassification.CrossValidate(cachedTrainData, learningPipelineLightGbmInductedToHallOfFame, 100,
            //    labelColumn: _labelColunmn, stratificationColumn: _labelColunmn);

            #endregion

            // Debug Only: view data pipeline data
            // var previewLearningPipeline = learningPipeline.Preview(cachedTrainData, 100, 100);

            Console.WriteLine("Finished Baseball Predictions - Model Job \n");

            #region Step 3) Report Metrics

            Console.WriteLine("##########################");
            Console.WriteLine("Step 4: Report Metrics...");
            Console.WriteLine("##########################\n");

            var labelColumns = new string[] { "OnHallOfFameBallot", "InductedToHallOfFame" };
            var algorithms = new string[] { "GeneralizedAdditiveModels", "LogisticRegression", "FastTree", "LightGbm",
                "StochasticDualCoordinateAscent", "StochasticGradientDescent" };


            for (int i = 0; i < algorithms.Length; i++)
            {
                for (int j = 0; j < labelColumns.Length; j++)
                {
                    var binaryClassificationMetrics = GetBinaryClassificationModelMetrics(labelColumns[j], algorithms[i], cachedValidationData);

                    Console.WriteLine("Evaluation Metrics for " + algorithms[i] + " | " + labelColumns[j]);
                    Console.WriteLine("******************");
                    Console.WriteLine("F1 Score:   " + Math.Round(binaryClassificationMetrics.F1Score, 4).ToString());
                    Console.WriteLine("AUC Score:  " + Math.Round(binaryClassificationMetrics.Auc, 4).ToString());
                    Console.WriteLine("Precision:  " + Math.Round(binaryClassificationMetrics.PositivePrecision, 4).ToString());
                    Console.WriteLine("Recall:     " + Math.Round(binaryClassificationMetrics.PositiveRecall, 4).ToString());
                    Console.WriteLine("Accuracy:   " + Math.Round(binaryClassificationMetrics.Accuracy, 4).ToString());
                    Console.WriteLine("******************");

                    var loadedModel = LoadModel(GetModelPath(algorithmName: algorithms[i], isOnnx: false, label: labelColumns[j]));
                    var transformedModelData = loadedModel.Transform(cachedTrainData);
                    TransformerChain<ITransformer> lastTran = (TransformerChain<ITransformer>) loadedModel.LastTransformer;
                    var enumerator = lastTran.GetEnumerator();

                    ISingleFeaturePredictionTransformer<IPredictorProducing<float>> transfomerForPfi = null;
                    while(enumerator.MoveNext())
                    {
                        if (enumerator.Current is BinaryPredictionTransformer<IPredictorProducing<float>>)
                        {
                            transfomerForPfi = enumerator.Current as ISingleFeaturePredictionTransformer<IPredictorProducing<float>>;
                        }
                    }

                    // TODO: FIX
                    // Retrieve Top Features based on Permutation Feature Importance
                    var permutationMetrics = _mlContext.BinaryClassification.PermutationFeatureImportance
                        (transfomerForPfi, transformedModelData, label: labelColumns[j], features: "Features", permutationCount: 10);

                    // Build a list of feature importance metrics
                    List<FeatureImportanceValue> featureImportanceValues = new List<FeatureImportanceValue>();
                    for (int k = 0; k < permutationMetrics.Length; k++)
                    {
                        featureImportanceValues.Add(
                                new FeatureImportanceValue
                                {
                                    FeatureName = featureColumns[k],
                                    PerformanceMetricName = "F1Score.Mean",
                                    PerformanceMetricValue = permutationMetrics[k].F1Score.Mean
                                }
                            );
                    }

                    // Filter out NaN values and order by lowest values
                    // Note: Should be done with absolute and check for positive values for features
                    var orderedFeatures = featureImportanceValues.Where(a => !Double.IsNaN(a.PerformanceMetricValue)).OrderBy(a => a.PerformanceMetricValue).ToList();
                    var numberOfFeaturesToReport = 4;

                    Console.WriteLine("Most important features (" + numberOfFeaturesToReport + ")");
                    Console.WriteLine("******************");

                    for (int l = 0; l < numberOfFeaturesToReport; l++)
                    {
                        if (l+1 <= featureImportanceValues.Count && l < orderedFeatures.Count)
                        {
                            Console.WriteLine(orderedFeatures[l].FeatureName + ": " + Math.Round(orderedFeatures[l].PerformanceMetricValue, 4).ToString());
                        }
                    }

                    Console.WriteLine("******************");
                    Console.WriteLine();
                }
            }



            #endregion

            #region Step 4) New Predictions - Using Ficticious Player Data

            Console.WriteLine("##########################");
            Console.WriteLine("Step 4: New Predictions...");
            Console.WriteLine("##########################\n");

            // Retrieve model path
            var algorithmTypeName = "LogisticRegression";
            var loadedModelOnHallOfFameBallot = LoadModel(GetModelPath(algorithmTypeName, false, "OnHallOfFameBallot"));
            var loadedModelInductedToHallOfFame = LoadModel(GetModelPath(algorithmTypeName, false, "InductedToHallOfFame"));

            // Create prediction engine
            var predEngineOnHallOfFameBallot = loadedModelOnHallOfFameBallot.CreatePredictionEngine<MLBBaseballBatter, MLBHOFPrediction>(_mlContext);
            var predEngineInductedToHallOfFame = loadedModelInductedToHallOfFame.CreatePredictionEngine<MLBBaseballBatter, MLBHOFPrediction>(_mlContext);

            // Create statistics for bad, average & great player
            var badMLBBatter = new MLBBaseballBatter
            {
                FullPlayerName = "Bad Player",
                ID = 100f,
                InductedToHallOfFame = false,
                LastYearPlayed = 0f,
                OnHallOfFameBallot = false,
                YearsPlayed = 2f,
                AB = 100f,
                R = 10f,
                H = 30f,
                Doubles = 1f,
                Triples = 1f,
                HR = 1f,
                RBI = 10f,
                SB = 10f,
                BattingAverage = 0.3f,
                SluggingPct = 0.15f,
                AllStarAppearances = 1f,
                MVPs = 0f,
                TripleCrowns = 0f,
                GoldGloves = 0f,
                MajorLeaguePlayerOfTheYearAwards = 0f,
                TB = 200f
            };
            var averageMLBBatter = new MLBBaseballBatter
            {
                FullPlayerName = "Average Player",
                ID = 100f,
                InductedToHallOfFame = false,
                LastYearPlayed = 0f,
                OnHallOfFameBallot = false,
                YearsPlayed = 2f,
                AB = 8393f,
                R = 1162f,
                H = 2340f,
                Doubles = 410f,
                Triples = 8f,
                HR = 439f,
                RBI = 1412f,
                SB = 9f,
                BattingAverage = 0.279f,
                SluggingPct = 0.486f,
                AllStarAppearances = 6f,
                MVPs = 0f,
                TripleCrowns = 0f,
                GoldGloves = 0f,
                MajorLeaguePlayerOfTheYearAwards = 0f,
                TB = 4083f
            };
            var greatMLBBatter = new MLBBaseballBatter
            {
                FullPlayerName = "Bad Player",
                ID = 100f,
                InductedToHallOfFame = false,
                LastYearPlayed = 0f,
                OnHallOfFameBallot = false,
                YearsPlayed = 20f,
                AB = 10000f,
                R = 1900f,
                H = 3000f,
                Doubles = 500f,
                Triples = 150f,
                HR = 600f,
                RBI = 1800f,
                SB = 400f,
                BattingAverage = 0.350f,
                SluggingPct = 0.65f,
                AllStarAppearances = 14f,
                MVPs = 2f,
                TripleCrowns = 1f,
                GoldGloves = 4f,
                MajorLeaguePlayerOfTheYearAwards = 2f,
                TB = 7000f
            };
            var batters = new List<MLBBaseballBatter> { badMLBBatter, averageMLBBatter, greatMLBBatter };
            // Convert the list to an IDataView
            var newPredictionsData = _mlContext.Data.ReadFromEnumerable(batters);

            // Make the predictions for both OnHallOfFameBallot & InductedToHallOfFame
            var predBadOnHallOfFameBallot = predEngineOnHallOfFameBallot.Predict(badMLBBatter);
            var predBadInductedToHallOfFame = predEngineInductedToHallOfFame.Predict(badMLBBatter);
            var predAverageOnHallOfFameBallot = predEngineOnHallOfFameBallot.Predict(averageMLBBatter);
            var predAverageInductedToHallOfFame = predEngineInductedToHallOfFame.Predict(averageMLBBatter);
            var predGreatOnHallOfFameBallot = predEngineOnHallOfFameBallot.Predict(greatMLBBatter);
            var predGreatInductedToHallOfFame = predEngineInductedToHallOfFame.Predict(greatMLBBatter);

            // Report the results
            Console.WriteLine("Algorithm Used for Model Prediction: " + algorithmTypeName);
            Console.WriteLine("Bad Baseball Player Prediction");
            Console.WriteLine("------------------------------");
            Console.WriteLine("On HOF Ballot Prediction: " + predBadOnHallOfFameBallot.Prediction.ToString() + " | " + "Probability: " + predBadOnHallOfFameBallot.Probability);
            Console.WriteLine("HOF Inducted Prediction:  " + predBadInductedToHallOfFame.Prediction.ToString() + " | " + "Probability: " + predBadInductedToHallOfFame.Probability);
            Console.WriteLine();
            Console.WriteLine("Average Baseball Player Prediction");
            Console.WriteLine("------------------------------");
            Console.WriteLine("On HOF Ballot Prediction: " + predAverageOnHallOfFameBallot.Prediction.ToString() + " | " + "Probability: " + predAverageOnHallOfFameBallot.Probability);
            Console.WriteLine("HOF Inducted Prediction:  " + predAverageInductedToHallOfFame.Prediction.ToString() + " | " + "Probability: " + predAverageInductedToHallOfFame.Probability);
            Console.WriteLine();
            Console.WriteLine("Great Baseball Player Prediction");
            Console.WriteLine("------------------------------");
            Console.WriteLine("On HOF Ballot Prediction: " + predGreatOnHallOfFameBallot.Prediction.ToString() + " | " + "Probability: " + predGreatOnHallOfFameBallot.Probability);
            Console.WriteLine("HOF Inducted Prediction:  " + predGreatInductedToHallOfFame.Prediction.ToString() + " | " + "Probability: " + predGreatInductedToHallOfFame.Probability);

            #endregion


            // TODO: FINISH

            //var loadedModelPath = GetModelPath("LightGbm", true, "OnHallOfFameBallot");
            //var session = new InferenceSession(loadedModelPath);
            //var inputInfo = session.InputMetadata.First();
            //var outputInfo = session.OutputMetadata.First();

            //VBuffer<float> weights = new VBuffer<float>();
            //modelLogisticRegressionInductedToHallOfFame.LastTransformer.Model.GetFeatureWeights(ref weights);

            //var transformedNewPredictionsData = modelLogisticRegressionInductedToHallOfFame.Transform(newPredictionsData);
            //var explainer = _mlContext.Model.Explainability.FeatureContributionCalculation(modelLogisticRegressionInductedToHallOfFame.LastTransformer.Model);
            //var outputData = explainer.Fit(transformedNewPredictionsData).Transform(transformedNewPredictionsData);

            //var scoringEnumerator = _mlContext.CreateEnumerable<BaseballBatterScoreAndFeatureContribution>(outputData, true).GetEnumerator();

            //int index = 0;
            //Console.WriteLine("Probability\tScore\tBiggestFeature      \t\tValue\tWeight\tContribution");
            //while (scoringEnumerator.MoveNext() && index < 4)
            //{
            //    var row = scoringEnumerator.Current;

            //    // Get the feature index with the biggest contribution
            //    var featureOfInterest = GetMostContributingFeature(row.FeatureContributions);

            //    // And the corresponding information about the feature
            //    var value = row.Features[featureOfInterest];
            //    var contribution = row.FeatureContributions[featureOfInterest];
            //    var name = featureColumns[featureOfInterest];
            //    var weight = weights.GetValues()[featureOfInterest];

            //    Console.WriteLine("{0:0.00}\t{1:0.00}\t\t{2}\t{3:0.00}\t{4:0.00}\t{5:0.00}",
            //        row.Probability,
            //        row.Score,
            //        name,
            //        value,
            //        weight,
            //        contribution
            //        );

            //    index++;
            //}

            // End of Job, report time
            Console.WriteLine();
            Console.WriteLine(string.Format("Model building job Finished in: {0} seconds", Math.Round(sw.Elapsed.TotalSeconds, 2)));
            Console.ReadLine();
        }

        /// <summary>
        /// Get Binary classfication metrics from a persisted model
        /// </summary>
        /// <param name="labelColumn"></param>
        /// <param name="algorithmTypeName"></param>
        /// <param name="validationData"></param>
        /// <returns></returns>
        private static CalibratedBinaryClassificationMetrics GetBinaryClassificationModelMetrics(string labelColumn, string algorithmTypeName, IDataView validationData)
        {
            // Retrieve model path
            var loadedModelPath = GetModelPath(algorithmTypeName, false, labelColumn);

            // Load model for both prediction types
            var loadedModel = LoadModel(loadedModelPath);

            var metrics = _mlContext.BinaryClassification.Evaluate(loadedModel.Transform(validationData), label: labelColumn);

            return metrics;
        }

        /// <summary>
        /// Get model path (ONNX, or MLNet (zip file)
        /// </summary>
        /// <param name="algorithmName"></param>
        /// <param name="isOnnx"></param>
        /// <param name="label"></param>
        /// <returns></returns>
        private static string GetModelPath(string algorithmName, bool isOnnx, string label)
        {
            string modelPathName = string.Empty;
            string modelName = string.Format("model-{0}-{1}.onnx", algorithmName, label);

            if (isOnnx)
            {
                modelPathName = Path.Combine(_appPath, "..", "..", "..", "Models", string.Format("model-{0}-{1}.onnx", algorithmName, label));
            }
            else
            {
                modelPathName = Path.Combine(_appPath, "..", "..", "..", "Models", string.Format("model-{0}-{1}.mlnet", algorithmName, label));
            }

            return modelPathName;
        }

        /// <summary>
        /// Gets a baseline pipeline used for all of the models
        /// </summary>
        /// <returns></returns>
        private static EstimatorChain<NormalizingTransformer> GetBaseLinePipeline()
        {
            var baselineTransform = _mlContext.Transforms.Concatenate("FeaturesBeforeNormalization", featureColumns)
                .Append(_mlContext.Transforms.Normalize("Features", "FeaturesBeforeNormalization",
                NormalizingEstimator.NormalizerMode.MinMax));

            // Debug Only: View transform
            // var pipelineTransformPreview = baselineTransform.Preview(dataTrain, 100);

            return baselineTransform;
        }

        /// <summary>
        /// Load a ML.NET model from a persisted file location
        /// </summary>
        /// <param name="loadedModelPath"></param>
        /// <returns></returns>
        private static TransformerChain<ITransformer> LoadModel(string loadedModelPath)
        {
            ITransformer loadedModel;
            using (var stream = File.OpenRead(loadedModelPath))
            {
                loadedModel = _mlContext.Model.Load(stream);
            }
            
            // Cast the loaded model to a transformer chain
            // This allows for interacting with the LastTransformer property
            TransformerChain<ITransformer> transfomerChain = (TransformerChain<ITransformer>) loadedModel;

            return transfomerChain;
        }

        /// <summary>
        /// Save a model using default ML.NET persistance
        /// </summary>
        /// <param name="algorithmName"></param>
        /// <param name="labelColumn"></param>
        /// <param name="model"></param>
        private static void SaveModel(string algorithmName, string labelColumn, ITransformer model)
        {
            var modelPath = GetModelPath(algorithmName, false, labelColumn);

            // Write out the model
            using (var fileStream = new FileStream(modelPath, FileMode.Create, FileAccess.Write, FileShare.Write))
            {
                _mlContext.Model.Save(model, fileStream);
            }
        }

        /// <summary>
        /// Save a model using ONNX persistance ML.NET transofrms
        /// Note: Use Netron Viewer to open up models
        /// </summary>
        /// <param name="algorithmName"></param>
        /// <param name="labelColumn"></param>
        /// <param name="model"></param>
        /// <param name="mlContext"></param>
        /// <param name="inputData"></param>
        private static void SaveOnnxModel(string algorithmName, string labelColumn, ITransformer model, MLContext mlContext, IDataView inputData)
        {
            var modelPath = GetModelPath(algorithmName, true, labelColumn);

            if (algorithmName != "AveragedPerceptron" && algorithmName != "GeneralizedAdditiveModels" && algorithmName != "LinearSupportVectorMachines")
            {
                var protoBufModel = mlContext.Model.ConvertToOnnx(model, inputData);

                // Write out the model (ONNX)
                using (var fileStream = new FileStream(modelPath, FileMode.Create, FileAccess.Write, FileShare.Write))
                {
                    protoBufModel.WriteTo(fileStream);
                }
            }
        }

        private static int GetMostContributingFeature(float[] featureContributions)
        {
            int index = 0;
            float currentValue = float.NegativeInfinity;
            for (int i = 0; i < featureContributions.Length; i++)
                if (featureContributions[i] > currentValue)
                {
                    currentValue = featureContributions[i];
                    index = i;
                }
            return index;
        }
    }
}
