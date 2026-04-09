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
        private const string FastTreeAlgorithmName = "FastTree";
        private const string LightGbmAlgorithmName = "LightGbm";
        private const int KeyLabelWidth = 28;

        private static readonly ConsoleColor StepHeaderColor = ConsoleColor.Yellow;
        private static readonly ConsoleColor JobStatusColor = ConsoleColor.White;
        private static readonly ConsoleColor NarrativeColor = ConsoleColor.Gray;
        private static readonly ConsoleColor SubsectionColor = ConsoleColor.Magenta;
        private static readonly ConsoleColor ActionColor = ConsoleColor.Cyan;
        private static readonly ConsoleColor SuccessColor = ConsoleColor.Green;
        private static readonly ConsoleColor WarningColor = ConsoleColor.DarkYellow;
        private static readonly ConsoleColor DangerColor = ConsoleColor.Red;
        private static readonly ConsoleColor KeyColor = ConsoleColor.DarkYellow;
        private static readonly ConsoleColor MetricValueColor = ConsoleColor.Cyan;
        private static readonly ConsoleColor StandardValueColor = ConsoleColor.White;
        private static readonly ConsoleColor SeparatorColor = ConsoleColor.DarkGray;
        private static readonly ConsoleColor FeatureColor = ConsoleColor.Gray;

        // Set up path locations
        private static string appFolder = Path.GetDirectoryName(Assembly.GetEntryAssembly().Location);
        private static string _trainDataPath => Path.Combine(appFolder, "Data", "MLBBaseballBattersSplitTraining.csv");
        private static string _testDataPath => Path.Combine(appFolder, "Data", "MLBBaseballBattersSplitTest.csv");
        private static string _fullDataPath => Path.Combine(appFolder, "Data", "MLBBaseballBattersFullTraining.csv");
        private static string _championChallengerMetrics => Path.Combine(appFolder, @"ModelPerformanceMetrics", "ChampionChallengerMetrics.csv");

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
            WriteColoredLine(JobStatusColor, "Starting Baseball Predictions - Training Model Job");
            WriteKeyValue("Using ML.NET Version:", "5.0");
            WriteKeyValue("Process Architecture:", processArchitecture);
            Console.WriteLine();
            WriteColoredLine(NarrativeColor, "This training job builds challenger/champion models across GAM, FastTree, and LightGbm that predict both:");
            WriteColoredLine(NarrativeColor, "1) Whether a baseball batter would make it on the HOF Ballot (OnHallOfFameBallot)");
            WriteColoredLine(NarrativeColor, "2) Whether a baseball batter would be inducted to the HOF (InductedToHallOfFame).");
            Console.WriteLine();
            WriteColoredLine(NarrativeColor, "The job executes a GAM hyperparameter sweep plus fixed FastTree and LightGbm challengers,");
            WriteColoredLine(NarrativeColor, "selects champions per algorithm and label, runs final holdout tests, and persists final full-data models.");
            Console.WriteLine();

            #region Step 1) ML.NET Setup & Load Data

            WriteStepHeader("Step 1: Load Data from files...");

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

            // Reset challenger/champion metrics output
            Directory.CreateDirectory(Path.GetDirectoryName(_championChallengerMetrics)!);
            File.Delete(_championChallengerMetrics);

            #endregion

            #region Step 2) Challenger/Champion Sweep

            WriteStepHeader("Step 2: Train Challengers and Pick Champions...");

            var challengerRuns = new List<CalibratedBinaryRunResult>();
            var finalChampionTestRuns = new List<CalibratedBinaryRunResult>();
            var algorithmHarnesses = GetAlgorithmHarnesses(processArchitecture);

            foreach (var labelColumn in labelColumns)
            {
                foreach (var harness in algorithmHarnesses)
                {
                    WriteColoredLine(SubsectionColor, $"Running {harness.AlgorithmName} challenger harness for label: {labelColumn}");

                    var labelRuns = TrainChallengersAndEvaluate(labelColumn, cachedTrainData, cachedTestData, harness);
                    RankChallengers(labelRuns);

                    var champion = labelRuns.First(r => r.Rank == 1);
                    champion.IsChampion = true;
                    challengerRuns.AddRange(labelRuns);

                    WriteColoredLine(SuccessColor, $"Champion selected for {labelColumn} | {harness.AlgorithmName} -> {champion.HyperparameterKey}");
                    WriteColoredLine(NarrativeColor, "Selection criteria: max AUPRC, tie-break max F1, then min LogLoss, then deterministic hyperparameter key");
                    Console.WriteLine();

                    var championConfig = harness.ChallengerConfigurations.Single(config => config.HyperparameterKey == champion.HyperparameterKey);
                    var championFinalTestRun = PromoteChampionAndEvaluate(
                        labelColumn,
                        harness,
                        championConfig,
                        champion,
                        cachedTrainData,
                        cachedTestData,
                        cachedFullData);

                    finalChampionTestRuns.Add(championFinalTestRun);

                    WriteColoredLine(SubsectionColor, $"Final Holdout Test for Champion | {labelColumn} | {harness.AlgorithmName}");
                    ReportRun(championFinalTestRun);
                }
            }

            Console.WriteLine();

            #endregion

            #region Step 3) Report Harness Metrics

            WriteStepHeader("Step 3: Report Harness Metrics...");

            var allHarnessRuns = challengerRuns
                .OrderBy(r => r.LabelColumn, StringComparer.Ordinal)
                .ThenBy(r => r.AlgorithmName, StringComparer.Ordinal)
                .ThenBy(r => r.RunType, StringComparer.Ordinal)
                .ThenBy(r => r.Rank)
                .ThenBy(r => r.HyperparameterKey, StringComparer.Ordinal)
                .Concat(finalChampionTestRuns
                    .OrderBy(r => r.LabelColumn, StringComparer.Ordinal)
                    .ThenBy(r => r.AlgorithmName, StringComparer.Ordinal))
                .ToList();
            // Keep report ordering deterministic so repeated runs are easy to diff/review.

            WriteMetricsCsv(_championChallengerMetrics, allHarnessRuns);

            WriteKeyValue("Challenger runs recorded:", challengerRuns.Count.ToString(CultureInfo.InvariantCulture));
            WriteKeyValue("Champion test runs:", finalChampionTestRuns.Count.ToString(CultureInfo.InvariantCulture));
            WriteKeyValue("Metrics CSV written to:", _championChallengerMetrics, FeatureColor);
            Console.WriteLine();

            #endregion

            #region Step 4) New Predictions - Using Fictitious Player Data

            WriteStepHeader("Step 4: New Predictions...");

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
            WriteKeyValue("Prediction algorithm:", GamAlgorithmName);
            Console.WriteLine();
            WriteColoredLine(NarrativeColor, $"Running predictions for {orderedSamples.Count} fictitious players (10 per tier).");
            Console.WriteLine();

            foreach (var sample in orderedSamples)
            {
                var predOnHallOfFameBallot = predEngineOnHallOfFameBallot.Predict(sample.Batter);
                var predInductedToHallOfFame = predEngineInductedToHallOfFame.Predict(sample.Batter);

                ReportPredictionSample(sample, predOnHallOfFameBallot, predInductedToHallOfFame);
            }

            #endregion

            // End of job, report time
            Console.WriteLine();
            WriteColoredLine(JobStatusColor, string.Format(CultureInfo.InvariantCulture, "Finished Baseball Predictions - Training Model Job in: {0} seconds", Math.Round(sw.Elapsed.TotalSeconds, 2)));
            Console.ReadLine();
        }

        private static IReadOnlyList<AlgorithmHarness> GetAlgorithmHarnesses(string processArchitecture)
        {
            var harnesses = new List<AlgorithmHarness>
            {
                new AlgorithmHarness
                {
                    AlgorithmName = GamAlgorithmName,
                    ChallengerConfigurations = BuildGamChallengerConfigurations()
                },
                new AlgorithmHarness
                {
                    AlgorithmName = FastTreeAlgorithmName,
                    ChallengerConfigurations = BuildFastTreeChallengerConfigurations()
                }
            };

            if (!string.Equals(processArchitecture, "Arm64", StringComparison.OrdinalIgnoreCase))
            {
                harnesses.Add(new AlgorithmHarness
                {
                    AlgorithmName = LightGbmAlgorithmName,
                    ChallengerConfigurations = BuildLightGbmChallengerConfigurations()
                });
            }
            else
            {
                WriteColoredLine(WarningColor, "LightGbm harness skipped on Arm64 because the native lib_lightgbm runtime is unavailable on this machine.");
                Console.WriteLine();
            }

            return harnesses;
        }

        private static IReadOnlyList<AlgorithmChallengerConfiguration> BuildGamChallengerConfigurations()
        {
            var configurations = new List<AlgorithmChallengerConfiguration>();

            foreach (var numberOfIterations in gamNumberOfIterations)
            {
                foreach (var maximumBinCountPerFeature in gamMaximumBinCountPerFeature)
                {
                    foreach (var learningRate in gamLearningRates)
                    {
                        var iterations = numberOfIterations;
                        var maximumBins = maximumBinCountPerFeature;
                        var rate = learningRate;

                        configurations.Add(new AlgorithmChallengerConfiguration
                        {
                            HyperparameterKey = string.Format(CultureInfo.InvariantCulture, "{0:D6}-{1:D4}-{2:0.000000}", iterations, maximumBins, rate),
                            HyperparameterSummary = string.Format(CultureInfo.InvariantCulture, "Iter={0} | MaxBin={1} | LearnRate={2:0.######}", iterations, maximumBins, rate),
                            CreateTrainer = labelColumn => new GamBaseballBatterTrainer(
                                labelColumn,
                                numberOfIterations: iterations,
                                maximumBinCountPerFeature: maximumBins,
                                learningRate: rate)
                        });
                    }
                }
            }

            return configurations;
        }

        private static IReadOnlyList<AlgorithmChallengerConfiguration> BuildFastTreeChallengerConfigurations()
        {
            return new List<AlgorithmChallengerConfiguration>
            {
                new AlgorithmChallengerConfiguration
                {
                    HyperparameterKey = "0020-0100-0010-0.200000",
                    HyperparameterSummary = "Leaves=20 | Trees=100 | MinLeaf=10 | LearnRate=0.2",
                    CreateTrainer = labelColumn => new FastTreeBaseballBatterTrainer(
                        labelColumn,
                        numberOfLeaves: 20,
                        numberOfTrees: 100,
                        minimumExampleCountPerLeaf: 10,
                        learningRate: 0.2)
                }
            };
        }

        private static IReadOnlyList<AlgorithmChallengerConfiguration> BuildLightGbmChallengerConfigurations()
        {
            return new List<AlgorithmChallengerConfiguration>
            {
                new AlgorithmChallengerConfiguration
                {
                    HyperparameterKey = "0100-default-default-default",
                    HyperparameterSummary = "Iter=100 | Leaves=default | MinLeaf=default | LearnRate=default",
                    CreateTrainer = labelColumn => new LightGbmBaseballBatterTrainer(
                        labelColumn,
                        numberOfLeaves: null,
                        minimumExampleCountPerLeaf: null,
                        learningRate: null,
                        numberOfIterations: 100)
                }
            };
        }

        private static List<CalibratedBinaryRunResult> TrainChallengersAndEvaluate(
            string labelColumn,
            IDataView trainData,
            IDataView testData,
            AlgorithmHarness harness)
        {
            var labelRuns = new List<CalibratedBinaryRunResult>();

            foreach (var configuration in harness.ChallengerConfigurations)
            {
                var trainer = configuration.CreateTrainer(labelColumn);

                WriteColoredLine(ActionColor, $"Training challenger: {trainer.Name}");
                trainer.Fit(trainData);

                // Save as test model path so it can be evaluated using calibrated metrics.
                // We evaluate from persisted model to match the same path used for downstream reporting.
                trainer.SaveModel(appFolder, false, trainData);

                var metrics = Utilities.GetBinaryClassificationModelMetrics(
                    isFinalModel: false,
                    appPath: appFolder,
                    mlContext: _mlContext,
                    labelColumn: labelColumn,
                    algorithmTypeName: harness.AlgorithmName,
                    validationData: testData);

                var run = CreateRunResult(
                    runType: "Challenger",
                    labelColumn: labelColumn,
                    algorithmName: harness.AlgorithmName,
                    hyperparameterKey: configuration.HyperparameterKey,
                    hyperparameterSummary: configuration.HyperparameterSummary,
                    metrics: metrics,
                    rank: 0,
                    isChampion: false);

                labelRuns.Add(run);
                ReportRun(run);
            }

            return labelRuns;
        }

        private static CalibratedBinaryRunResult PromoteChampionAndEvaluate(
            string labelColumn,
            AlgorithmHarness harness,
            AlgorithmChallengerConfiguration championConfiguration,
            CalibratedBinaryRunResult champion,
            IDataView trainData,
            IDataView testData,
            IDataView fullData)
        {
            // This artifact represents the selected challenger evaluated on the holdout split.
            var championTestTrainer = championConfiguration.CreateTrainer(labelColumn);
            championTestTrainer.Fit(trainData);
            championTestTrainer.SaveModel(appFolder, false, trainData);

            var championFinalTestMetrics = Utilities.GetBinaryClassificationModelMetrics(
                isFinalModel: false,
                appPath: appFolder,
                mlContext: _mlContext,
                labelColumn: labelColumn,
                algorithmTypeName: harness.AlgorithmName,
                validationData: testData);

            var championFinalTestRun = CreateRunResult(
                runType: "ChampionFinalTest",
                labelColumn: labelColumn,
                algorithmName: harness.AlgorithmName,
                hyperparameterKey: champion.HyperparameterKey,
                hyperparameterSummary: champion.HyperparameterSummary,
                metrics: championFinalTestMetrics,
                rank: 1,
                isChampion: true);

            // Final models are always retrained on full data after selection so inference uses all rows.
            var championFinalTrainer = championConfiguration.CreateTrainer(labelColumn);
            championFinalTrainer.Fit(fullData);
            championFinalTrainer.SaveModel(appFolder, true, fullData);

            return championFinalTestRun;
        }

        private static void RankChallengers(List<CalibratedBinaryRunResult> labelRuns)
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

        private static CalibratedBinaryRunResult CreateRunResult(
            string runType,
            string labelColumn,
            string algorithmName,
            string hyperparameterKey,
            string hyperparameterSummary,
            CalibratedBinaryClassificationMetrics metrics,
            int rank,
            bool isChampion)
        {
            // Flatten metrics into a single row object so console and CSV reporting share the same source.
            var (tp, tn, fp, fn) = GetConfusionCounts(metrics);

            return new CalibratedBinaryRunResult
            {
                RunType = runType,
                LabelColumn = labelColumn,
                AlgorithmName = algorithmName,
                Seed = seed,
                Rank = rank,
                IsChampion = isChampion,
                HyperparameterKey = hyperparameterKey,
                HyperparameterSummary = hyperparameterSummary,
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

        private static void ReportRun(CalibratedBinaryRunResult run)
        {
            WriteSeparator('*', 34, SubsectionColor);
            WriteKeyValue("Run Type:", run.RunType);
            WriteKeyValue("Label:", run.LabelColumn);
            WriteKeyValue("Algorithm:", run.AlgorithmName);
            WriteKeyValue("Hyperparameters:", run.HyperparameterSummary);
            WriteKeyValue("Hyperparameter Key:", run.HyperparameterKey);
            WriteKeyValue("Rank:", run.Rank.ToString(CultureInfo.InvariantCulture), run.Rank == 1 ? SuccessColor : StandardValueColor);
            WriteKeyValue("Champion:", run.IsChampion.ToString(), run.IsChampion ? SuccessColor : StandardValueColor);
            WriteSeparator('*', 34);
            WriteKeyValue("F1 Score:", FormatMetric(run.F1Score), MetricValueColor);
            WriteKeyValue("AUC - Prec/Recall:", FormatMetric(run.AreaUnderPrecisionRecallCurve), MetricValueColor);
            WriteKeyValue("AUC - ROC:", FormatMetric(run.AreaUnderRocCurve), MetricValueColor);
            WriteKeyValue("Positive Precision:", FormatMetric(run.PositivePrecision), MetricValueColor);
            WriteKeyValue("Positive Recall:", FormatMetric(run.PositiveRecall), MetricValueColor);
            WriteKeyValue("Negative Precision:", FormatMetric(run.NegativePrecision), MetricValueColor);
            WriteKeyValue("Negative Recall:", FormatMetric(run.NegativeRecall), MetricValueColor);
            WriteKeyValue("Accuracy:", FormatMetric(run.Accuracy), MetricValueColor);
            WriteKeyValue("LogLoss:", FormatMetric(run.LogLoss), MetricValueColor);
            WriteKeyValue("LogLoss Reduction:", FormatMetric(run.LogLossReduction), MetricValueColor);
            WriteKeyValue("Entropy:", FormatMetric(run.Entropy), MetricValueColor);
            WriteKeyValue(
                "TP/TN/FP/FN:",
                string.Format(
                    CultureInfo.InvariantCulture,
                    "{0}/{1}/{2}/{3}",
                    Math.Round(run.TruePositiveCount, 0),
                    Math.Round(run.TrueNegativeCount, 0),
                    Math.Round(run.FalsePositiveCount, 0),
                    Math.Round(run.FalseNegativeCount, 0)),
                MetricValueColor);
            WriteSeparator('*', 34);
            Console.WriteLine();
        }

        private static void ReportPredictionSample(
            PlayerPredictionSample sample,
            MLBHOFPrediction predOnHallOfFameBallot,
            MLBHOFPrediction predInductedToHallOfFame)
        {
            WriteColoredLine(GetTierColor(sample.Tier), $"{sample.DisplayName} ({sample.Batter.ID}) - Tier: {sample.Tier}");
            WriteSeparator('-', 50);
            WritePredictionResult("On HOF Ballot:", predOnHallOfFameBallot);
            WriteKeyValue("Ballot top features:", MachineLearning.Utilities.GetTopContributingFeatures(predOnHallOfFameBallot), FeatureColor);
            WritePredictionResult("HOF Inducted:", predInductedToHallOfFame);
            WriteKeyValue("Inducted top features:", MachineLearning.Utilities.GetTopContributingFeatures(predInductedToHallOfFame), FeatureColor);
            Console.WriteLine();
        }

        private static void WritePredictionResult(string label, MLBHOFPrediction prediction)
        {
            var predictionColor = prediction.Prediction ? SuccessColor : DangerColor;
            var predictionSummary = string.Format(
                CultureInfo.InvariantCulture,
                "{0} | Probability: {1}",
                prediction.Prediction,
                FormatMetric(prediction.Probability));

            WriteKeyValue(label, predictionSummary, predictionColor);
        }

        private static string FormatMetric(double value)
        {
            return Math.Round(value, 6).ToString("0.######", CultureInfo.InvariantCulture);
        }

        private static ConsoleColor GetTierColor(PlayerPredictionTier tier)
        {
            switch (tier)
            {
                case PlayerPredictionTier.Bad:
                    return DangerColor;
                case PlayerPredictionTier.Great:
                    return SuccessColor;
                default:
                    return StandardValueColor;
            }
        }

        private static void WriteStepHeader(string title)
        {
            var border = new string('#', Math.Max(title.Length, 31));
            WriteColoredLine(StepHeaderColor, border);
            WriteColoredLine(StepHeaderColor, title);
            WriteColoredLine(StepHeaderColor, border);
            Console.WriteLine();
        }

        private static void WriteSeparator(char separatorCharacter, int length, ConsoleColor? color = null)
        {
            WriteColoredLine(color ?? SeparatorColor, new string(separatorCharacter, length));
        }

        private static void WriteKeyValue(string label, string value, ConsoleColor? valueColor = null)
        {
            Console.ForegroundColor = KeyColor;
            Console.Write(label.PadRight(KeyLabelWidth));
            Console.ForegroundColor = valueColor ?? StandardValueColor;
            Console.WriteLine(value);
            Console.ResetColor();
        }

        private static void WriteColoredLine(ConsoleColor color, string message)
        {
            Console.ForegroundColor = color;
            Console.WriteLine(message);
            Console.ResetColor();
        }

        private static void WriteMetricsCsv(string outputPath, IReadOnlyCollection<CalibratedBinaryRunResult> runs)
        {
            var header = string.Join(",",
                "RunType",
                "LabelColumn",
                "AlgorithmName",
                "Seed",
                "Rank",
                "IsChampion",
                "HyperparameterKey",
                "HyperparameterSummary",
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
                        run.HyperparameterKey,
                        run.HyperparameterSummary,
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

        private class AlgorithmHarness
        {
            public string AlgorithmName { get; set; }
            public IReadOnlyList<AlgorithmChallengerConfiguration> ChallengerConfigurations { get; set; }
        }

        private class AlgorithmChallengerConfiguration
        {
            public string HyperparameterKey { get; set; }
            public string HyperparameterSummary { get; set; }
            public Func<string, ITrainerBase> CreateTrainer { get; set; }
        }

        private class CalibratedBinaryRunResult
        {
            public string RunType { get; set; }
            public string LabelColumn { get; set; }
            public string AlgorithmName { get; set; }
            public int Seed { get; set; }
            public int Rank { get; set; }
            public bool IsChampion { get; set; }
            public string HyperparameterKey { get; set; }
            public string HyperparameterSummary { get; set; }
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
