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

        public string Name { get; protected set; }

        protected static string ModelPath => Path
                          .Combine(AppContext.BaseDirectory, "classification.mdl");

        protected readonly MLContext _mlContext;

        protected string[] _featureColumns;

        protected DataOperationsCatalog.TrainTestData _dataSplit;
        protected ITrainerEstimator<BinaryPredictionTransformer<TParameters>, TParameters> _model;
        protected ITransformer _trainedModel;

        protected BaseballBatterTrainerBase()
        {
            _mlContext = new MLContext(100);
        }

        /// <summary>
        /// Train model on defined data.
        /// </summary>
        /// <param name="trainingFileName"></param>
        public void Fit(string trainingFileName)
        {
            if (!File.Exists(trainingFileName))
            {
                throw new FileNotFoundException($"Training file {trainingFileName} doesn't exist.");
            }

            _dataSplit = LoadAndPrepareData(trainingFileName);
            var dataProcessPipeline = GetBaseLinePipeline();
            var trainingPipeline = dataProcessPipeline.Append(_model);

            _trainedModel = trainingPipeline.Fit(_dataSplit.TrainSet);
        }

        /// <summary>
        /// Evaluate trained model.
        /// </summary>
        /// <returns>Model performance.</returns>
        public BinaryClassificationMetrics Evaluate()
        {
            var testSetTransform = _trainedModel.Transform(_dataSplit.TestSet);

            return _mlContext.BinaryClassification.EvaluateNonCalibrated(testSetTransform);
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
            var baselineTransform = _mlContext.Transforms.Concatenate("FeaturesBeforeNormalization", _featureColumns)
                .Append(_mlContext.Transforms.NormalizeMinMax("Features", "FeaturesBeforeNormalization"));

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
