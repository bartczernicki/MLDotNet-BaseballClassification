using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLDotNet_BaseballClassification.MachineLearning
{
    public class Utilities
    {
        public static string[] FeatureColumns = new string[] {
            "YearsPlayed", "AB", "R", "H", "Doubles", "Triples", "HR", "RBI", "SB",
            "BattingAverage", "SluggingPct", "AllStarAppearances", "TB", "TotalPlayerAwards"
            // Other Features
            /*, "MVPs", "TripleCrowns", "GoldGloves", "MajorLeaguePlayerOfTheYearAwards"*/
        };

        public static Microsoft.ML.Data.EstimatorChain<Microsoft.ML.Data.ColumnConcatenatingTransformer> GetBaseLinePipeline(Microsoft.ML.MLContext mlContext, string[] featureColumns)
        {
            var baselineTransform = mlContext.Transforms.Concatenate("FeaturesBeforeNormalization", featureColumns)
                .Append(mlContext.Transforms.NormalizeMinMax("Features", "FeaturesBeforeNormalization"))
                .Append(mlContext.Transforms.Concatenate("Features", featureColumns));

            return baselineTransform;
        }

        public static PredictionEngine<MLBBaseballBatter, MLBHOFPrediction> CreatePredictionEngine(
            MLContext mlContext, TransformerChain<ITransformer> trainedModel, IDataView cachedData)
        {
            var baselineTransform = Utilities.GetBaseLinePipeline(mlContext, FeatureColumns);

            var newDataTransformer = baselineTransform.Fit(cachedData);
            var singleRow = newDataTransformer.Transform(mlContext.Data.TakeRows(cachedData, 1));

            var singleFeatureTransformer
                = trainedModel.LastTransformer as Microsoft.ML.Data.BinaryPredictionTransformer<Microsoft.ML.Calibrators.CalibratedModelParametersBase<Microsoft.ML.Trainers.FastTree.GamBinaryModelParameters, Microsoft.ML.Calibrators.PlattCalibrator>>;

            var contributionEstimator = mlContext.Transforms
            .CalculateFeatureContribution(
                predictionTransformer: singleFeatureTransformer,
                numberOfPositiveContributions: FeatureColumns.Length,
                numberOfNegativeContributions: FeatureColumns.Length,
                normalize: false)
            .Fit(singleRow);

            // Create the full transformer chain.
            var scoringPipeline = trainedModel
                .Append(contributionEstimator);

            var predictionEngineWithFeatureContribution = mlContext.Model.CreatePredictionEngine<MLBBaseballBatter, MLBHOFPrediction>(scoringPipeline);

            return predictionEngineWithFeatureContribution;
        }

        public static string GetTopContributingFeatures(MLBHOFPrediction prediction, int topCount = 3)
        {
            if (prediction.Probability > 0.5f)
            {
                var topContributions = prediction.FeatureContributions
                    .Select((value, index) => new { Value = value, Index = index })
                    .OrderByDescending(x => x.Value)
                    .Take(topCount)
                    .Select(x => new { Feature = Utilities.FeatureColumns[x.Index], Contribution = x.Value })
                    .ToList();

                // Return the names concatenated by a comma
                return string.Join(", ", topContributions.Select(x => $"{x.Feature}"));
            }
            else
            {
                var topContributions = prediction.FeatureContributions
                    .Select((value, index) => new { Value = value, Index = index })
                    .OrderBy(x => x.Value)
                    .Take(topCount)
                    .Select(x => new { Feature = Utilities.FeatureColumns[x.Index], Contribution = x.Value })
                    .ToList();

                // Return the names concatenated by a comma
                return string.Join(", ", topContributions.Select(x => $"{x.Feature}"));
            }

        }
    }
}
