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

        public string AlgorithmName { get; protected set; }

        public string LabelColumnName { get; protected set; }

        public string Name { get; protected set; }

        protected readonly MLContext _mlContext;

        protected DataViewSchema dataSchema;
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

            // Set the schema
            this.dataSchema = trainingData.Schema;

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
        public void SaveModel(string folderPath, bool isOnnx, bool isFinalModel)
        {
            var modelPath = GetModelPath(folderPath, isOnnx, false);

            // Write out the model
            using (var fileStream = new FileStream(modelPath, FileMode.Create, FileAccess.Write, FileShare.Write))
            {
                this._mlContext.Model.Save(_trainedModel, this.dataSchema, fileStream);
            }
        }

        private EstimatorChain<NormalizingTransformer> GetBaseLinePipeline()
        {
            // Build baseline platform and cache
            var baselineTransform = _mlContext.Transforms.Concatenate("FeaturesBeforeNormalization", _featureColumns)
                .Append(_mlContext.Transforms.NormalizeMinMax("Features", "FeaturesBeforeNormalization"))
                .AppendCacheCheckpoint(this._mlContext);

            return baselineTransform;
        }

        public string GetModelPath(string folderPath, bool isOnnx, bool isFinalModel)
        {
            var modelPrefix = this.LabelColumnName.Replace("HallOfFame", "HoF");

            // Model persistance convention used:
            // model + algorithmName + dependent variable column name + model persistance type extension (ONNX or native ML.NET)
            string modelPathName = string.Empty;
            string modelName = string.Format("{0}-{1}.onnx", modelPrefix, this.AlgorithmName);
            string modelFolder = isFinalModel ? "Final" : "Test";

            if (isOnnx)
            {
                modelPathName = Path.Combine(folderPath, $@"Models\{modelFolder}", string.Format("{0}-{1}.onnx", modelPrefix, this.AlgorithmName));
            }
            else
            {
                modelPathName = Path.Combine(folderPath, $@"Models\{modelFolder}", string.Format("{0}-{1}.mlnet", modelPrefix, this.AlgorithmName));
            }

            return modelPathName;
        }
    }
}
