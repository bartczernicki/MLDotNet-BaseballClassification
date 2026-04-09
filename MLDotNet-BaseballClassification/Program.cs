using Microsoft.ML;
using Microsoft.ML.Data;
using MLDotNet_BaseballClassification.MachineLearning;
using MLDotNet_BaseballClassification.MachineLearning.Trainers;
using MLDotNet_BaseballClassification.Services;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Reflection;

namespace MLDotNet_BaseballClassification
{
    class Program
    {
        private const string GamAlgorithmName = "GeneralizedAdditiveModels";

        // Set up path locations
        private static string appFolder = Path.GetDirectoryName(Assembly.GetEntryAssembly().Location);
        private static string _trainDataPath => Path.Combine(appFolder, "Data", "MLBBaseballBattersSplitTraining.csv");
        private static string _testDataPath => Path.Combine(appFolder, "Data", "MLBBaseballBattersSplitTest.csv");
        private static string _fullDataPath => Path.Combine(appFolder, "Data", "MLBBaseballBattersFullTraining.csv");
        private static string _gamChampionChallengerMetrics => Path.Combine(appFolder, @"ModelPerformanceMetrics", "GamChampionChallengerMetrics.csv");

        // Thread-safe ML Context
        private static MLContext _mlContext;
        // Set seed to static value for re-producible model results (or DateTime for pseudo-random)
        private static int seed = 100;

        // List of supervised learning labels
        // Use: At least one must be left
        private static string[] labelColumns = new string[] { "OnHallOfFameBallot", "InductedToHallOfFame" };

        // GAM hyperparameter sweep (medium grid: 27 combinations per label)
        private static int[] gamNumberOfIterations = new int[] { 10000, 50000, 100000 };
        private static int[] gamMaximumBinCountPerFeature = new int[] { 64, 128, 255 };
        private static double[] gamLearningRates = new double[] { 0.001, 0.002, 0.005 };

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
            Console.WriteLine("Using ML.NET - Version 5.0");
            Console.WriteLine("Process Architecture: {0}", processArchitecture);
            Console.WriteLine();
            Console.ResetColor();
            Console.WriteLine("This training job builds GAM challenger/champion models that predict both:");
            Console.WriteLine("1) Whether a baseball batter would make it on the HOF Ballot (OnHallOfFameBallot)");
            Console.WriteLine("2) Whether a baseball batter would be inducted to the HOF (InductedToHallOfFame).\n");
            Console.WriteLine("The job executes a GAM hyperparameter sweep, selects a champion per label,");
            Console.WriteLine("runs final holdout tests for each champion, and persists final full-data models.\n");

            #region Step 1) ML.NET Setup & Load Data

            Console.ForegroundColor = ConsoleColor.Yellow;
            Console.WriteLine("###############################");
            Console.WriteLine("Step 1: Load Data from files...");
            Console.WriteLine("###############################\n");
            Console.ResetColor();

            // Set the seed explicitly for reproducibility (models will be built with consistent results)
            _mlContext = new MLContext(seed: seed);

            var dataTrain = _mlContext.Data.LoadFromTextFile<MLBBaseballBatter>(path: _trainDataPath,
                hasHeader: true, separatorChar: ',', allowQuoting: false);
            var dataTest = _mlContext.Data.LoadFromTextFile<MLBBaseballBatter>(path: _testDataPath,
                hasHeader: true, separatorChar: ',', allowQuoting: false);
            var dataFull = _mlContext.Data.LoadFromTextFile<MLBBaseballBatter>(path: _fullDataPath,
                hasHeader: true, separatorChar: ',', allowQuoting: false);

            // Cache the loaded data
            var cachedTrainData = _mlContext.Data.Cache(dataTrain);
            var cachedTestData = _mlContext.Data.Cache(dataTest);
            var cachedFullData = _mlContext.Data.Cache(dataFull);
            // The job intentionally keeps both split data (model selection/evaluation) and full data
            // (final retrain for inference artifacts) in memory to avoid repeated file reads.

            // Reset GAM champion/challenger metrics output
            Directory.CreateDirectory(Path.GetDirectoryName(_gamChampionChallengerMetrics)!);
            File.Delete(_gamChampionChallengerMetrics);

            #endregion

            #region Step 2) GAM Challenger/Champion Sweep

            Console.ForegroundColor = ConsoleColor.Yellow;
            Console.WriteLine("##############################################");
            Console.WriteLine("Step 2: Train GAM Challengers and Pick Champions...");
            Console.WriteLine("##############################################\n");
            Console.ResetColor();

            var challengerRuns = new List<GamRunResult>();
            var finalChampionTestRuns = new List<GamRunResult>();
            var championByLabel = new Dictionary<string, GamRunResult>(StringComparer.Ordinal);

            foreach (var labelColumn in labelColumns)
            {
                Console.ForegroundColor = ConsoleColor.Cyan;
                Console.WriteLine($"Running GAM challenger sweep for label: {labelColumn}");
                Console.ResetColor();

                var labelRuns = TrainGamChallengersAndEvaluate(labelColumn, cachedTrainData, cachedTestData);
                RankGamChallengers(labelRuns);

                var champion = labelRuns.First(r => r.Rank == 1);
                champion.IsChampion = true;
                championByLabel[labelColumn] = champion;
                challengerRuns.AddRange(labelRuns);

                Console.ForegroundColor = ConsoleColor.Green;
                Console.WriteLine($"Champion selected for {labelColumn} -> {champion.HyperparameterKey}");
                Console.WriteLine($"Selection criteria: max AUPRC, tie-break max F1, then min LogLoss, then deterministic hyperparameter key");
                Console.ResetColor();
                Console.WriteLine();

                // Persist champion TEST model (trained on split training set)
                // This artifact represents the selected challenger evaluated on the holdout split.
                var championTestTrainer = new GamBaseballBatterTrainer(
                    labelColumn,
                    numberOfIterations: champion.NumberOfIterations,
                    maximumBinCountPerFeature: champion.MaximumBinCountPerFeature,
                    learningRate: champion.LearningRate);

                championTestTrainer.Fit(cachedTrainData);
                championTestTrainer.SaveModel(appFolder, false, cachedTrainData);

                // Final holdout test for selected champion
                var championFinalTestMetrics = Utilities.GetBinaryClassificationModelMetrics(
                    isFinalModel: false,
                    appPath: appFolder,
                    mlContext: _mlContext,
                    labelColumn: labelColumn,
                    algorithmTypeName: GamAlgorithmName,
                    validationData: cachedTestData);

                var championFinalTestRun = CreateGamRunResult(
                    runType: "ChampionFinalTest",
                    labelColumn: labelColumn,
                    numberOfIterations: champion.NumberOfIterations,
                    maximumBinCountPerFeature: champion.MaximumBinCountPerFeature,
                    learningRate: champion.LearningRate,
                    metrics: championFinalTestMetrics,
                    rank: 1,
                    isChampion: true);

                finalChampionTestRuns.Add(championFinalTestRun);

                Console.ForegroundColor = ConsoleColor.Yellow;
                Console.WriteLine($"Final Holdout Test for Champion | {labelColumn}");
                Console.ResetColor();
                ReportGamRun(championFinalTestRun);

                // Persist champion FINAL model (re-trained on full data)
                // Final models are always retrained on full data after selection so inference uses all rows.
                var championFinalTrainer = new GamBaseballBatterTrainer(
                    labelColumn,
                    numberOfIterations: champion.NumberOfIterations,
                    maximumBinCountPerFeature: champion.MaximumBinCountPerFeature,
                    learningRate: champion.LearningRate);

                championFinalTrainer.Fit(cachedFullData);
                championFinalTrainer.SaveModel(appFolder, true, cachedFullData);
            }

            Console.WriteLine();

            #endregion

            #region Step 3) Report GAM Metrics

            Console.ForegroundColor = ConsoleColor.Yellow;
            Console.WriteLine("###############################");
            Console.WriteLine("Step 3: Report GAM Metrics...");
            Console.WriteLine("###############################\n");
            Console.ResetColor();

            var allGamRuns = challengerRuns
                .OrderBy(r => r.LabelColumn, StringComparer.Ordinal)
                .ThenBy(r => r.RunType, StringComparer.Ordinal)
                .ThenBy(r => r.Rank)
                .ThenBy(r => r.HyperparameterKey, StringComparer.Ordinal)
                .Concat(finalChampionTestRuns.OrderBy(r => r.LabelColumn, StringComparer.Ordinal))
                .ToList();
            // Keep report ordering deterministic so repeated runs are easy to diff/review.

            WriteGamMetricsCsv(_gamChampionChallengerMetrics, allGamRuns);

            Console.WriteLine($"Challenger runs recorded: {challengerRuns.Count}");
            Console.WriteLine($"Champion final-test runs recorded: {finalChampionTestRuns.Count}");
            Console.WriteLine($"Metrics CSV written to: {_gamChampionChallengerMetrics}");
            Console.WriteLine();

            #endregion

            #region Step 4) New Predictions - Using Fictitious Player Data

            Console.ForegroundColor = ConsoleColor.Yellow;
            Console.WriteLine("###############################");
            Console.WriteLine("Step 4: New Predictions...");
            Console.WriteLine("###############################\n");
            Console.ResetColor();

            // Retrieve final champion model paths
            // Prediction/demo step always uses the final full-data GAM models saved above.
            var loadedModelOnHallOfFameBallot = Utilities.LoadModel(_mlContext, (Utilities.GetModelPath(appFolder, GamAlgorithmName, false, "OnHallOfFameBallot", true)));
            var loadedModelInductedToHallOfFame = Utilities.LoadModel(_mlContext, (Utilities.GetModelPath(appFolder, GamAlgorithmName, false, "InductedToHallOfFame", true)));

            // Create prediction engine with Feature Contribution support
            var predEngineOnHallOfFameBallot = MachineLearning.Utilities.CreatePredictionEngine(_mlContext, loadedModelOnHallOfFameBallot, cachedFullData);
            var predEngineInductedToHallOfFame = MachineLearning.Utilities.CreatePredictionEngine(_mlContext, loadedModelInductedToHallOfFame, cachedFullData);

            var playerSampleService = new FictitiousPlayerSampleService();
            var playerSamples = playerSampleService.GetSamples();

            if (playerSamples.Count != 30)
            {
                throw new InvalidOperationException($"Expected exactly 30 fictitious player samples, but found {playerSamples.Count}.");
            }

            var badSampleCount = playerSamples.Count(s => s.Tier == PlayerPredictionTier.Bad);
            var averageSampleCount = playerSamples.Count(s => s.Tier == PlayerPredictionTier.Average);
            var greatSampleCount = playerSamples.Count(s => s.Tier == PlayerPredictionTier.Great);

            if (badSampleCount != 10 || averageSampleCount != 10 || greatSampleCount != 10)
            {
                throw new InvalidOperationException(
                    $"Expected 10 samples per tier, but found Bad={badSampleCount}, Average={averageSampleCount}, Great={greatSampleCount}.");
            }

            var orderedSamples = playerSamples
                .OrderBy(s => s.Tier)
                .ThenBy(s => s.DisplayName, StringComparer.Ordinal)
                .ToList();

            // Report the results
            Console.WriteLine("Algorithm Used for sample Model Prediction: " + GamAlgorithmName);
            Console.WriteLine("\n");
            Console.WriteLine($"Running predictions for {orderedSamples.Count} fictitious players (10 per tier).\n");

            foreach (var sample in orderedSamples)
            {
                var predOnHallOfFameBallot = predEngineOnHallOfFameBallot.Predict(sample.Batter);
                var predInductedToHallOfFame = predEngineInductedToHallOfFame.Predict(sample.Batter);

                Console.WriteLine($"{sample.DisplayName} ({sample.Batter.ID}) - Tier: {sample.Tier}");
                Console.WriteLine("--------------------------------------------------");
                Console.WriteLine("On HOF Ballot Prediction: " + predOnHallOfFameBallot.Prediction + " | Probability: " + predOnHallOfFameBallot.Probability);
                Console.WriteLine("On HOF Ballot Prediction - Top Contributing Features: " + MachineLearning.Utilities.GetTopContributingFeatures(predOnHallOfFameBallot));
                Console.WriteLine("HOF Inducted Prediction:  " + predInductedToHallOfFame.Prediction + " | Probability: " + predInductedToHallOfFame.Probability);
                Console.WriteLine("HOF Inducted Prediction - Top Contributing Features: " + MachineLearning.Utilities.GetTopContributingFeatures(predInductedToHallOfFame));
                Console.WriteLine();
            }

            #endregion

            // End of job, report time
            Console.WriteLine();
            Console.ForegroundColor = ConsoleColor.Cyan;
            Console.WriteLine(string.Format("Finished Baseball Predictions - Training Model Job in: {0} seconds", Math.Round(sw.Elapsed.TotalSeconds, 2)));
            Console.ReadLine();
        }

        private static List<GamRunResult> TrainGamChallengersAndEvaluate(string labelColumn, IDataView trainData, IDataView testData)
        {
            var labelRuns = new List<GamRunResult>();

            foreach (var numberOfIterations in gamNumberOfIterations)
            {
                foreach (var maximumBinCountPerFeature in gamMaximumBinCountPerFeature)
                {
                    foreach (var learningRate in gamLearningRates)
                    {
                        var trainer = new GamBaseballBatterTrainer(
                            labelColumn,
                            numberOfIterations: numberOfIterations,
                            maximumBinCountPerFeature: maximumBinCountPerFeature,
                            learningRate: learningRate);

                        Console.WriteLine($"Training GAM challenger: {trainer.Name}");
                        trainer.Fit(trainData);

                        // Save as test model path so it can be evaluated using calibrated metrics.
                        // We evaluate from persisted model to match the same path used for downstream reporting.
                        trainer.SaveModel(appFolder, false, trainData);

                        var metrics = Utilities.GetBinaryClassificationModelMetrics(
                            isFinalModel: false,
                            appPath: appFolder,
                            mlContext: _mlContext,
                            labelColumn: labelColumn,
                            algorithmTypeName: GamAlgorithmName,
                            validationData: testData);

                        var run = CreateGamRunResult(
                            runType: "Challenger",
                            labelColumn: labelColumn,
                            numberOfIterations: numberOfIterations,
                            maximumBinCountPerFeature: maximumBinCountPerFeature,
                            learningRate: learningRate,
                            metrics: metrics,
                            rank: 0,
                            isChampion: false);

                        labelRuns.Add(run);
                        ReportGamRun(run);
                    }
                }
            }

            return labelRuns;
        }

        private static void RankGamChallengers(List<GamRunResult> labelRuns)
        {
            // Champion ranking priority:
            // 1) maximize AUPRC, 2) maximize F1, 3) minimize LogLoss, 4) deterministic key ordering.
            var orderedRuns = labelRuns
                .OrderByDescending(r => r.AreaUnderPrecisionRecallCurve)
                .ThenByDescending(r => r.F1Score)
                .ThenBy(r => r.LogLoss)
                .ThenBy(r => r.HyperparameterKey, StringComparer.Ordinal)
                .ToList();

            for (int i = 0; i < orderedRuns.Count; i++)
            {
                orderedRuns[i].Rank = i + 1;
                orderedRuns[i].IsChampion = (i == 0);
            }
        }

        private static GamRunResult CreateGamRunResult(
            string runType,
            string labelColumn,
            int numberOfIterations,
            int maximumBinCountPerFeature,
            double learningRate,
            CalibratedBinaryClassificationMetrics metrics,
            int rank,
            bool isChampion)
        {
            // Flatten metrics into a single row object so console and CSV reporting share the same source.
            var (tp, tn, fp, fn) = GetConfusionCounts(metrics);

            return new GamRunResult
            {
                RunType = runType,
                LabelColumn = labelColumn,
                AlgorithmName = GamAlgorithmName,
                Seed = seed,
                NumberOfIterations = numberOfIterations,
                MaximumBinCountPerFeature = maximumBinCountPerFeature,
                LearningRate = learningRate,
                Rank = rank,
                IsChampion = isChampion,
                F1Score = metrics.F1Score,
                AreaUnderPrecisionRecallCurve = metrics.AreaUnderPrecisionRecallCurve,
                AreaUnderRocCurve = metrics.AreaUnderRocCurve,
                PositivePrecision = metrics.PositivePrecision,
                PositiveRecall = metrics.PositiveRecall,
                NegativePrecision = metrics.NegativePrecision,
                NegativeRecall = metrics.NegativeRecall,
                Accuracy = metrics.Accuracy,
                LogLoss = metrics.LogLoss,
                LogLossReduction = metrics.LogLossReduction,
                Entropy = metrics.Entropy,
                TruePositiveCount = tp,
                TrueNegativeCount = tn,
                FalsePositiveCount = fp,
                FalseNegativeCount = fn
            };
        }

        private static (double truePositiveCount, double trueNegativeCount, double falsePositiveCount, double falseNegativeCount)
            GetConfusionCounts(CalibratedBinaryClassificationMetrics metrics)
        {
            var counts = metrics.ConfusionMatrix.Counts;
            if (counts.Count < 2 || counts[0].Count < 2 || counts[1].Count < 2)
            {
                return (0d, 0d, 0d, 0d);
            }

            // For binary classification:
            // Rows are actual labels [negative, positive], columns are predicted [negative, positive].
            var trueNegativeCount = counts[0][0];
            var falsePositiveCount = counts[0][1];
            var falseNegativeCount = counts[1][0];
            var truePositiveCount = counts[1][1];

            return (truePositiveCount, trueNegativeCount, falsePositiveCount, falseNegativeCount);
        }

        private static void ReportGamRun(GamRunResult run)
        {
            Console.WriteLine($"RunType:                   {run.RunType}");
            Console.WriteLine($"Label:                     {run.LabelColumn}");
            Console.WriteLine($"Hyperparameters:           Iter={run.NumberOfIterations}, MaxBin={run.MaximumBinCountPerFeature}, LearnRate={run.LearningRate}");
            Console.WriteLine($"Hyperparameter Key:        {run.HyperparameterKey}");
            Console.WriteLine($"Rank:                      {run.Rank}");
            Console.WriteLine($"Champion:                  {run.IsChampion}");
            Console.WriteLine("**************************");
            Console.WriteLine("F1 Score:                  " + Math.Round(run.F1Score, 6));
            Console.WriteLine("AUC - Prec/Recall Score:   " + Math.Round(run.AreaUnderPrecisionRecallCurve, 6));
            Console.WriteLine("AUC - ROC Score:           " + Math.Round(run.AreaUnderRocCurve, 6));
            Console.WriteLine("Positive Precision:        " + Math.Round(run.PositivePrecision, 6));
            Console.WriteLine("Positive Recall:           " + Math.Round(run.PositiveRecall, 6));
            Console.WriteLine("Negative Precision:        " + Math.Round(run.NegativePrecision, 6));
            Console.WriteLine("Negative Recall:           " + Math.Round(run.NegativeRecall, 6));
            Console.WriteLine("Accuracy:                  " + Math.Round(run.Accuracy, 6));
            Console.WriteLine("LogLoss:                   " + Math.Round(run.LogLoss, 6));
            Console.WriteLine("LogLossReduction:          " + Math.Round(run.LogLossReduction, 6));
            Console.WriteLine("Entropy:                   " + Math.Round(run.Entropy, 6));
            Console.WriteLine("TP/TN/FP/FN:               " +
                $"{Math.Round(run.TruePositiveCount, 0)}/{Math.Round(run.TrueNegativeCount, 0)}/{Math.Round(run.FalsePositiveCount, 0)}/{Math.Round(run.FalseNegativeCount, 0)}");
            Console.WriteLine("**************************");
            Console.WriteLine();
        }

        private static void WriteGamMetricsCsv(string outputPath, IReadOnlyCollection<GamRunResult> runs)
        {
            var header = string.Join(",",
                "RunType",
                "LabelColumn",
                "AlgorithmName",
                "Seed",
                "Rank",
                "IsChampion",
                "NumberOfIterations",
                "MaximumBinCountPerFeature",
                "LearningRate",
                "HyperparameterKey",
                "F1Score",
                "AreaUnderPrecisionRecallCurve",
                "AreaUnderRocCurve",
                "PositivePrecision",
                "PositiveRecall",
                "NegativePrecision",
                "NegativeRecall",
                "Accuracy",
                "LogLoss",
                "LogLossReduction",
                "Entropy",
                "TruePositiveCount",
                "TrueNegativeCount",
                "FalsePositiveCount",
                "FalseNegativeCount");

            using (var writer = File.AppendText(outputPath))
            {
                writer.WriteLine(header);

                foreach (var run in runs)
                {
                    // Use invariant formatting so decimal separators are stable across locales.
                    var row = string.Join(",",
                        run.RunType,
                        run.LabelColumn,
                        run.AlgorithmName,
                        run.Seed.ToString(CultureInfo.InvariantCulture),
                        run.Rank.ToString(CultureInfo.InvariantCulture),
                        run.IsChampion.ToString(),
                        run.NumberOfIterations.ToString(CultureInfo.InvariantCulture),
                        run.MaximumBinCountPerFeature.ToString(CultureInfo.InvariantCulture),
                        run.LearningRate.ToString("0.######", CultureInfo.InvariantCulture),
                        run.HyperparameterKey,
                        run.F1Score.ToString("0.###############", CultureInfo.InvariantCulture),
                        run.AreaUnderPrecisionRecallCurve.ToString("0.###############", CultureInfo.InvariantCulture),
                        run.AreaUnderRocCurve.ToString("0.###############", CultureInfo.InvariantCulture),
                        run.PositivePrecision.ToString("0.###############", CultureInfo.InvariantCulture),
                        run.PositiveRecall.ToString("0.###############", CultureInfo.InvariantCulture),
                        run.NegativePrecision.ToString("0.###############", CultureInfo.InvariantCulture),
                        run.NegativeRecall.ToString("0.###############", CultureInfo.InvariantCulture),
                        run.Accuracy.ToString("0.###############", CultureInfo.InvariantCulture),
                        run.LogLoss.ToString("0.###############", CultureInfo.InvariantCulture),
                        run.LogLossReduction.ToString("0.###############", CultureInfo.InvariantCulture),
                        run.Entropy.ToString("0.###############", CultureInfo.InvariantCulture),
                        run.TruePositiveCount.ToString("0.###############", CultureInfo.InvariantCulture),
                        run.TrueNegativeCount.ToString("0.###############", CultureInfo.InvariantCulture),
                        run.FalsePositiveCount.ToString("0.###############", CultureInfo.InvariantCulture),
                        run.FalseNegativeCount.ToString("0.###############", CultureInfo.InvariantCulture));

                    writer.WriteLine(row);
                }
            }
        }

        private class GamRunResult
        {
            public string RunType { get; set; }
            public string LabelColumn { get; set; }
            public string AlgorithmName { get; set; }
            public int Seed { get; set; }
            public int Rank { get; set; }
            public bool IsChampion { get; set; }
            public int NumberOfIterations { get; set; }
            public int MaximumBinCountPerFeature { get; set; }
            public double LearningRate { get; set; }
            public string HyperparameterKey =>
                string.Format(CultureInfo.InvariantCulture, "{0:D6}-{1:D4}-{2:0.000000}", NumberOfIterations, MaximumBinCountPerFeature, LearningRate);

            public double F1Score { get; set; }
            public double AreaUnderPrecisionRecallCurve { get; set; }
            public double AreaUnderRocCurve { get; set; }
            public double PositivePrecision { get; set; }
            public double PositiveRecall { get; set; }
            public double NegativePrecision { get; set; }
            public double NegativeRecall { get; set; }
            public double Accuracy { get; set; }
            public double LogLoss { get; set; }
            public double LogLossReduction { get; set; }
            public double Entropy { get; set; }
            public double TruePositiveCount { get; set; }
            public double TrueNegativeCount { get; set; }
            public double FalsePositiveCount { get; set; }
            public double FalseNegativeCount { get; set; }
        }
    }
}
