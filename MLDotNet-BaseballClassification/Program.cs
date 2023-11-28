using Microsoft.ML;
using Microsoft.ML.Trainers;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Reflection;
using System.Linq;
using MathNet.Numerics;
using MathNet.Numerics.Statistics;
using MLDotNet_BaseballClassification.MachineLearning;
using MLDotNet_BaseballClassification.MachineLearning.Trainers;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace MLDotNet_BaseballClassification
{
    class Program
    {
        // Set up path locations
        private static string appFolder = Path.GetDirectoryName(Assembly.GetEntryAssembly().Location);
        private static string _trainDataPath => Path.Combine(appFolder, "Data", "MLBBaseballBattersSplitTraining.csv");
        private static string _testDataPath => Path.Combine(appFolder, "Data", "MLBBaseballBattersSplitTest.csv");
        private static string _fullDataPath => Path.Combine(appFolder, "Data", "MLBBaseballBattersFullTraining.csv");
        private static string _performanceMetricsTrainTestModels => Path.Combine(appFolder, @"ModelPerformanceMetrics", "PerformanceMetricsTrainTestModels.csv");

        // Thread-safe ML Context
        private static MLContext _mlContext;
        // Set seed to static value for re-producable model results (or DateTime for pseudo-random)
        private static int seed = 100;

        // Configuration Arrays

        // List of feature columns used for training
        // Useage: Comment out (or uncomment) feature names in order to explicitly select features for model training
        private static string[] featureColumns = new string[] {
            "YearsPlayed", "AB", "R", "H", "Doubles", "Triples", "HR", "RBI", "SB",
            "BattingAverage", "SluggingPct", "AllStarAppearances", "TB", "TotalPlayerAwards"
            // Other Features
            /*, "MVPs", "TripleCrowns", "GoldGloves", "MajorLeaguePlayerOfTheYearAwards"*/
        };
        private static string featureColumnsStringArray = String.Join(",", featureColumns);

        // List of supervised learning labels
        // Useage: At least one must be left
        private static string[] labelColumns = new string[] { "OnHallOfFameBallot", "InductedToHallOfFame" };

        // List of algorithms that support probability output
        // Useage: Comment out (or uncomment) algorithm names to report model explainability
        private static List<string> algorithmsForModelExplainability = new List<string> {
                "LogisticRegression",
                "FastTree", /*"LightGbm",*/
                "StochasticGradientDescentCalibrated",
                "GeneralizedAdditiveModels"
        };

        static void Main(string[] args)
        {
            // Check Processor Architecture (LightGBM)
            var processArchitecture = System.Runtime.InteropServices.RuntimeInformation.ProcessArchitecture.ToString();

            // Start stopwatch to time model job
            Stopwatch sw = new Stopwatch();
            sw.Start();

            Console.Title = "Baseball Predictions - Training Model Job";
            Console.ForegroundColor = ConsoleColor.Cyan;
            Console.WriteLine("Starting Baseball Predictions - Training Model Job");
            Console.WriteLine("Using ML.NET - Version 3.0");
            Console.WriteLine("Process Architecture: {0}", processArchitecture);
            Console.WriteLine();
            Console.ResetColor();
            Console.WriteLine("This training job will build a series of models that will predict both:");
            Console.WriteLine("1) Whether a baseball batter would make it on the HOF Ballot (OnHallOfFameBallot)");
            Console.WriteLine("2) Whether a baseball batter would be inducted to the HOF (InductedToHallOfFame).");
            Console.WriteLine("Based on an MLB batter's summarized career batting statistics (complete 2021 season).\n");
            Console.WriteLine("Note: The goal is to build a 'good enough' set of models & showcase the ML.NET framework.");
            Console.WriteLine("Note: For better models advanced historical scaling and features should be performed.");
            Console.WriteLine();

            #region Step 1) ML.NET Setup & Load Data

            Console.ForegroundColor = ConsoleColor.Yellow;
            Console.WriteLine("###############################");
            Console.WriteLine("Step 1: Load Data from files...");
            Console.WriteLine("###############################\n");
            Console.ResetColor();

            // Set the seed explicitly for reproducability (models will be built with consistent results)
            _mlContext = new MLContext(seed: seed);

            // Read the training/validation data from a text file
            //var dataTrainBatters = File.ReadAllLines(_trainDataPath)
            //    .Skip(1) // Skip the CSV Header
            //    .Select(v => MLBBaseballBatter.FromCsv(v))
            //    .AsQueryable() // Allows for Dyanmic Linq
            //    .Select("new (" + featureColumnsStringArray + ")")
            //    .ToDynamicList();

            var dataTrain = _mlContext.Data.LoadFromTextFile<MLBBaseballBatter>(path: _trainDataPath,
                hasHeader: true, separatorChar: ',', allowQuoting: false);
            var dataTest = _mlContext.Data.LoadFromTextFile<MLBBaseballBatter>(path: _testDataPath,
                hasHeader: true, separatorChar: ',', allowQuoting: false);
            var dataFull = _mlContext.Data.LoadFromTextFile<MLBBaseballBatter>(path: _fullDataPath,
                hasHeader: true, separatorChar: ',', allowQuoting: false);

            // TODO: REMOVE
            //dynamic myDynamic = new { PropertyOne = true, PropertyTwo = false };
            //var test = myDynamic.GetType();
            //var dynamicList = new List<dynamic>();
            //dynamicList.Add(myDynamic);
            //dynamicList.Add(myDynamic);
            //var test2 = _mlContext.Data.LoadFromEnumerable<dynamic>(dynamicList);
            //var pre = test2.Preview();
            //var testD = dataTrainBatters.FirstOrDefault();
            //Microsoft.ML.Data.SchemaDefinition sd;
            //var schemaDefinition = SchemaDefinition.Create(testD.GetType());
            //var test2 = _mlContext.Data.LoadFromEnumerable<dynamic>(dataTrainBatters, schemaDefinition );
            //var test2preview = test2.Preview();

            // Retrieve Data Schema
            var dataSchema = dataTrain.Schema;

            #if DEBUG
            // Debug Only: Preview the training/test data
            var dataTrainPreview = dataTrain.Preview();
            var dataTestPreview = dataTest.Preview();
            var dataFullPreview = dataFull.Preview();
            #endif

            // Cache the loaded data
            var cachedTrainData = _mlContext.Data.Cache(dataTrain);
            var cachedTestData = _mlContext.Data.Cache(dataTest);
            var cachedFullData = _mlContext.Data.Cache(dataFull);

            // Delete the Performance Metrics File(s)
            File.Delete(_performanceMetricsTrainTestModels);

            #endregion

            #region Step 2) Build Multiple Machine Learning Models

            // Notes:
            // Model training is for demo purposes and uses the default hyperparameters.
            // Default parameters were used in optimizing for large data sets.
            // It is best practice to always provide hyperparameters explicitly in order to have historical reproducability
            // as the ML.NET API evolves.

            Console.ForegroundColor = ConsoleColor.Yellow;
            Console.WriteLine("###############################");
            Console.WriteLine("Step 2: Train Models...");
            Console.WriteLine("###############################\n");
            Console.ResetColor();

            // Build list of BaseballBatter Trainers
            var trainers = new List<ITrainerBase>();


            foreach(var labelColumn in labelColumns)
            {
                // Do not perform LightGBM over Arm64
                // Add it to the list of algorithms for model explainability if x64
                if (processArchitecture != "Arm64")
                {
                    trainers.Add(new LightGbmBaseballBatterTrainer(labelColumn, numberOfIterations: 5000, learningRate: 0.002));
                    algorithmsForModelExplainability.Add("LightGbm");
                }
                
                trainers.Add(new AveragedPerceptronBaseballBatterTrainer(labelColumn, numberOfIterations: 1000));
                trainers.Add(new FastForestBaseballBatterTrainer(labelColumn, numberOfTrees: 500, numberOfLeaves: 50));
                trainers.Add(new FastTreeBaseballBatterTrainer(labelColumn, numberOfLeaves: 50, numberOfTrees: 500, learningRate: 0.002));
                trainers.Add(new GamBaseballBatterTrainer(labelColumn, numberOfIterations: 50000, learningRate: 0.001));
                trainers.Add(new LinearSvmBaseballBatterTrainer(labelColumn, numberOfIterations: 1000));
                trainers.Add(new LbfgsLogisticRegressionBaseballBatterTrainer(labelColumn, l1Regularization: 0.9f, l2Regularization: 0.9f));
                trainers.Add(new SgdCalibratedBaseballBatterTrainer(labelColumn, numberOfIterations: 1000, learningRate: 0.002));
                trainers.Add(new SgdNonCalibratedBaseballBatterTrainer(labelColumn, numberOfIterations: 1000, learningRate: 0.002));
            };

            foreach(var trainer in trainers)
            {
                // Fit a trainer on training data & evaluate performance metrics
                Console.WriteLine($"Training...{trainer.Name} Test model.");
                trainer.Fit(cachedTrainData);
                var performanceMetrics = trainer.Evaluate(cachedTestData);
                // Save model
                trainer.SaveModel(appFolder, false, cachedTrainData);

                // Fit a trainer on full data & persist final model
                Console.WriteLine($"Training...{trainer.Name} Final model.");
                trainer.Fit(cachedTrainData);
                // Save model
                trainer.SaveModel(appFolder, true, cachedFullData);
            }

            Console.WriteLine(string.Empty);

            #endregion

            //#region Step 3) Cross-Validate GAM

            //Console.ForegroundColor = ConsoleColor.Yellow;
            //Console.WriteLine("###############################");
            //Console.WriteLine("Step 3: Cross Validate GAM");
            //Console.WriteLine("###############################\n");
            //Console.ResetColor();

            //Console.WriteLine("Cross Validating GeneralizedAdditiveModels");

            //var crossValidationPerformance = _mlContext.BinaryClassification.CrossValidate(cachedFullData, learningPipelineGeneralizedAdditiveModelsOnHallOfFameBallot, 20,
            //    labelColumnName: "OnHallOfFameBallot", seed: seed);

            //Console.WriteLine("Accuracy");
            //crossValidationPerformance.Select(fold => fold.Metrics.Accuracy).ToList().ForEach(i => Console.Write("{0},", Math.Round(i, 4)));
            //var accuracyStdDev = crossValidationPerformance.Select(fold => fold.Metrics.Accuracy).ToList().StandardDeviation();
            //var accuracyMean = crossValidationPerformance.Select(fold => fold.Metrics.Accuracy).ToList().Mean();
            //var accuracyConfidenceRangeLower = Math.Round(accuracyMean - 1.96*accuracyStdDev, 3);
            //var accuracyConfidenceRangeHigher = Math.Round(accuracyMean + 1.96*accuracyStdDev, 3);
            //Console.WriteLine(string.Empty);
            //Console.WriteLine("Accuracy: " + accuracyMean + " 95% CI: " + accuracyConfidenceRangeLower + " - " + accuracyConfidenceRangeHigher);

            //Console.WriteLine("F1Score");
            //crossValidationPerformance.Select(fold => fold.Metrics.F1Score).ToList().ForEach(i => Console.Write("{0},", Math.Round(i, 4)));
            //var f1ScoreStdDev = crossValidationPerformance.Select(fold => fold.Metrics.F1Score).ToList().StandardDeviation();
            //var f1ScoreMean = crossValidationPerformance.Select(fold => fold.Metrics.F1Score).ToList().Mean();
            //var f1ScoreConfidenceRangeLower = Math.Round(f1ScoreMean - 1.96 * f1ScoreStdDev, 3);
            //var f1ScoreConfidenceRangeHigher = Math.Round(f1ScoreMean + 1.96 * f1ScoreStdDev, 3);
            //Console.WriteLine(string.Empty);
            //Console.WriteLine("F1Score: " + f1ScoreMean + "95% CI: " + f1ScoreConfidenceRangeLower + " - " + f1ScoreConfidenceRangeHigher);

            //Console.WriteLine("PositivePrecision");
            //crossValidationPerformance.Select(fold => fold.Metrics.PositivePrecision).ToList().ForEach(i => Console.Write("{0},", Math.Round(i, 4)));
            //var positivePrecisionStdDev = crossValidationPerformance.Select(fold => fold.Metrics.PositivePrecision).ToList().StandardDeviation();
            //var positivePrecisionMean = crossValidationPerformance.Select(fold => fold.Metrics.PositivePrecision).ToList().Mean();
            //var positivePrecisionRangeLower = Math.Round(positivePrecisionMean - 1.96 * positivePrecisionStdDev, 3);
            //var positivePrecisionRangeHigher = Math.Round(positivePrecisionMean + 1.96 * positivePrecisionStdDev, 3);
            //Console.WriteLine(string.Empty);
            //Console.WriteLine("PositivePrecision: " + positivePrecisionMean + " 95% CI: " + positivePrecisionRangeLower + " - " + positivePrecisionRangeHigher);

            //Console.WriteLine("PositiveRecall");
            //crossValidationPerformance.Select(fold => fold.Metrics.PositiveRecall).ToList().ForEach(i => Console.Write("{0},", Math.Round(i, 4)));
            //var positiveRecallStdDev = crossValidationPerformance.Select(fold => fold.Metrics.PositiveRecall).ToList().StandardDeviation();
            //var positiveRecallMean = crossValidationPerformance.Select(fold => fold.Metrics.PositiveRecall).ToList().Mean();
            //var positiveRecallRangeLower = Math.Round(positiveRecallMean - 1.96 * positiveRecallStdDev, 3);
            //var positiveRecallRangeHigher = Math.Round(positiveRecallMean + 1.96 * positiveRecallStdDev, 3);
            //Console.WriteLine(string.Empty);
            //Console.WriteLine("PositiveRecall: " + positiveRecallMean + " 95% CI: " + positiveRecallRangeLower + " - " + positiveRecallRangeHigher);

            //#endregion

            #region Step 4) Report Performance Metrics

            Console.ForegroundColor = ConsoleColor.Yellow;
            Console.WriteLine("###############################");
            Console.WriteLine("Step 4: Report Metrics...");
            Console.WriteLine("###############################\n");
            Console.ResetColor();

            // Write the performance metrics HEADER
            var performanceMetricsTrainTestHeaderRow = $@"{"AlgorithmName"},{"LabelColumn"},{"Seed"},{"F1Score"},{"AreaUnderPrecisionRecallCurve"},{"AreaUnderRocCurve"},{"PositivePrecision"},{"PositiveRecall"},{"Accuracy"},{"LogLoss"}";
            using (System.IO.StreamWriter file = File.AppendText(_performanceMetricsTrainTestModels))
            {
                file.WriteLine(performanceMetricsTrainTestHeaderRow);
            }

            for (int i = 0; i < algorithmsForModelExplainability.Count(); i++)
            {
                for (int j = 0; j < labelColumns.Length; j++)
                {
                    // TRAIN/TEST MODEL PERFORMANCE METRICS
                    var isFinalModel = false;
                    var binaryClassificationMetrics = Utilities.GetBinaryClassificationModelMetrics(isFinalModel, appFolder, _mlContext, labelColumns[j], algorithmsForModelExplainability[i], cachedTestData);

                    var metricF1Score = Math.Round(binaryClassificationMetrics.F1Score, 4);
                    var metricAreaUnderPrecisionRecallCurve = Math.Round(binaryClassificationMetrics.AreaUnderPrecisionRecallCurve, 4);
                    var metricAreaUnderRocCurve = Math.Round(binaryClassificationMetrics.AreaUnderRocCurve, 4);
                    var metricPositivePrecision = Math.Round(binaryClassificationMetrics.PositivePrecision, 4);
                    var metricPositiveRecall = Math.Round(binaryClassificationMetrics.PositiveRecall, 4);
                    var metricAccuracy = Math.Round(binaryClassificationMetrics.Accuracy, 4);
                    var metricLogLoss = Math.Round(binaryClassificationMetrics.LogLoss, 4);

                    Console.WriteLine("TRAIN/TEST Performance Metrics for " + algorithmsForModelExplainability[i] + " | " + labelColumns[j]);
                    Console.WriteLine("**************************");
                    Console.WriteLine("F1 Score:                 " + metricF1Score);
                    Console.WriteLine("AUC - Prec/Recall Score:  " + metricAreaUnderPrecisionRecallCurve);
                    Console.WriteLine("AUC - ROC Score:          " + metricAreaUnderRocCurve);
                    Console.WriteLine("Precision:                " + metricPositivePrecision);
                    Console.WriteLine("Recall:                   " + metricPositiveRecall);
                    Console.WriteLine("Accuracy:                 " + metricAccuracy);
                    Console.WriteLine("LogLoss:                  " + metricLogLoss);
                    Console.WriteLine("**************************");

                    // Write the performance metrics to file
                    var performanceMetricsTrainTestRow = $@"{algorithmsForModelExplainability[i]},{labelColumns[j]},{seed},{metricF1Score},{metricAreaUnderPrecisionRecallCurve},{metricAreaUnderRocCurve},{metricPositivePrecision},{metricPositiveRecall},{metricAccuracy},{metricLogLoss}";
                    using (System.IO.StreamWriter file = File.AppendText(_performanceMetricsTrainTestModels))
                    {
                        file.WriteLine(performanceMetricsTrainTestRow);
                    }

                    Console.WriteLine("**************************");
                    Console.WriteLine();
                }
            }



            #endregion

            #region Step 5) New Predictions - Using Ficticious Player Data

            Console.ForegroundColor = ConsoleColor.Yellow;
            Console.WriteLine("###############################");
            Console.WriteLine("Step 5: New Predictions...");
            Console.WriteLine("###############################\n");
            Console.ResetColor();

            // Set algorithm type to use for predictions
            // Retrieve model path
            // TODO: Hardcoded add perscriptive rules engine
            var algorithmTypeName = "GeneralizedAdditiveModels";
            var loadedModelOnHallOfFameBallot = Utilities.LoadModel(_mlContext, (Utilities.GetModelPath(appFolder, algorithmTypeName, false, "OnHallOfFameBallot", true)));
            var loadedModelInductedToHallOfFame = Utilities.LoadModel(_mlContext, (Utilities.GetModelPath(appFolder, algorithmTypeName, false, "InductedToHallOfFame", true)));

            // Create prediction engine
            var predEngineOnHallOfFameBallot = _mlContext.Model.CreatePredictionEngine<MLBBaseballBatter, MLBHOFPrediction>(loadedModelOnHallOfFameBallot);
            var predEngineInductedToHallOfFame = _mlContext.Model.CreatePredictionEngine<MLBBaseballBatter, MLBHOFPrediction>(loadedModelInductedToHallOfFame);

            // Create statistics for bad, average & great player
            var badMLBBatter = new MLBBaseballBatter
            {
                FullPlayerName = "Bad Player",
                ID = "Bad101",
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
                //MVPs = 0f,
                //TripleCrowns = 0f,
                //GoldGloves = 0f,
                //MajorLeaguePlayerOfTheYearAwards = 0f,
                TB = 200f
            };
            var averageMLBBatter = new MLBBaseballBatter
            {
                FullPlayerName = "Average Player",
                ID = "Avg101",
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
                //MVPs = 0f,
                //TripleCrowns = 0f,
                //GoldGloves = 0f,
                //MajorLeaguePlayerOfTheYearAwards = 0f,
                TB = 4083f
            };
            var greatMLBBatter = new MLBBaseballBatter
            {
                FullPlayerName = "Great Player",
                ID = "Great101",
                InductedToHallOfFame = false,
                LastYearPlayed = 0f,
                OnHallOfFameBallot = false,
                YearsPlayed = 20f,
                AB = 10000f,
                R = 1900f,
                H = 3500f,
                Doubles = 500f,
                Triples = 150f,
                HR = 600f,
                RBI = 1800f,
                SB = 400f,
                BattingAverage = 0.350f,
                SluggingPct = 0.65f,
                AllStarAppearances = 14f,
                //MVPs = 2f,
                //TripleCrowns = 1f,
                //GoldGloves = 4f,
                //MajorLeaguePlayerOfTheYearAwards = 2f,
                TB = 7000f
            };

            var batters = new List<MLBBaseballBatter> { badMLBBatter, averageMLBBatter, greatMLBBatter };
            // Convert the list to an IDataView
            var newPredictionsData = _mlContext.Data.LoadFromEnumerable(batters);

            // Make the predictions for both OnHallOfFameBallot & InductedToHallOfFame
            var predBadOnHallOfFameBallot = predEngineOnHallOfFameBallot.Predict(badMLBBatter);
            var predBadInductedToHallOfFame = predEngineInductedToHallOfFame.Predict(badMLBBatter);
            var predAverageOnHallOfFameBallot = predEngineOnHallOfFameBallot.Predict(averageMLBBatter);
            var predAverageInductedToHallOfFame = predEngineInductedToHallOfFame.Predict(averageMLBBatter);
            var predGreatOnHallOfFameBallot = predEngineOnHallOfFameBallot.Predict(greatMLBBatter);
            var predGreatInductedToHallOfFame = predEngineInductedToHallOfFame.Predict(greatMLBBatter);

            // Report the results
            Console.WriteLine("Algorithm Used for sample Model Prediction: " + algorithmTypeName);
            Console.WriteLine("\n");
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

            // End of job, report time
            Console.WriteLine();
            Console.ForegroundColor = ConsoleColor.Cyan;
            Console.WriteLine(string.Format("Finished Baseball Predictions - Training Model Job in: {0} seconds", Math.Round(sw.Elapsed.TotalSeconds, 2)));
            Console.ReadLine();
        }
    }
}
