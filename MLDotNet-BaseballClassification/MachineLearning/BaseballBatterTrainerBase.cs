using Microsoft.ML;
using Microsoft.ML.Calibrators;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
using System;
using System.IO;

namespace MLDotNet_BaseballClassification.MachineLearning
{
    /// <summary>
    /// Base class for Trainers.
    /// This class exposes methods for training, evaluating and saving ML Models.
    /// Classes that inherit this class need to assing concrete model and name.
    /// </summary>
    public abstract class BaseballBatterTrainerBase<TParameters> : ITrainerBase
        where TParameters : class
    {
        private int _seed = 100;

        private static string[] _featureColumns = new string[] {
            "YearsPlayed", "AB", "R", "H", "Doubles", "Triples", "HR", "RBI", "SB",
            "BattingAverage", "SluggingPct", "AllStarAppearances", "TB", "TotalPlayerAwards"
            // Other Features
            /*, "MVPs", "TripleCrowns", "GoldGloves", "MajorLeaguePlayerOfTheYearAwards"*/
        };

        public string LabelColumnName { get; protected set; }

        public string Name { get; protected set; }

        protected static string ModelPath => Path
                          .Combine(AppContext.BaseDirectory, "classification.mdl");

        protected readonly MLContext _mlContext;



        protected DataOperationsCatalog.TrainTestData _dataSplit;
        protected ITrainerEstimator<BinaryPredictionTransformer<TParameters>, TParameters> _trainerEstimator;
        protected ITransformer _trainedModel;

        protected BaseballBatterTrainerBase()
        {
            _mlContext = new MLContext(_seed);
        }

        /// <summary>
        /// Train model on defined data.
        /// </summary>
        /// <param name="trainingFileName"></param>
        public void Fit(IDataView trainingData)
        {
            //if (!File.Exists(trainingFileName))
            //{
            //    throw new FileNotFoundException($"Training file {trainingFileName} doesn't exist.");
            //}

            var dataProcessPipeline = GetBaseLinePipeline();
            var trainingPipeline = dataProcessPipeline.Append(_trainerEstimator);

            _trainedModel = trainingPipeline.Fit(trainingData);
        }

        /// <summary>
        /// Evaluate trained model.
        /// </summary>
        /// <returns>Model performance.</returns>
        public BinaryClassificationMetrics Evaluate(IDataView testData)
        {
            var testSetTransform = _trainedModel.Transform(testData);

            return _mlContext.BinaryClassification.Evaluate(testSetTransform, labelColumnName: this.LabelColumnName);
        }

        /// <summary>
        /// Save Model in the file.
        /// </summary>
        public void SaveModel()
        {
            _mlContext.Model.Save(_trainedModel, _dataSplit.TrainSet.Schema, ModelPath);
        }

        private EstimatorChain<NormalizingTransformer> GetBaseLinePipeline()
        {
            // Build baseline platform and cache
            var baselineTransform = _mlContext.Transforms.Concatenate("FeaturesBeforeNormalization", _featureColumns)
                .Append(_mlContext.Transforms.NormalizeMinMax("Features", "FeaturesBeforeNormalization"))
                .AppendCacheCheckpoint(this._mlContext);

            return baselineTransform;
        }

        private DataOperationsCatalog.TrainTestData LoadAndPrepareData(string trainingFileName)
        {
            var trainingDataView = _mlContext.Data
                                    .LoadFromTextFile<MLBBaseballBatter>
                                      (trainingFileName, hasHeader: true, separatorChar: ',');

            return _mlContext.Data.TrainTestSplit(trainingDataView, testFraction: 0.3);
        }
    }
}
