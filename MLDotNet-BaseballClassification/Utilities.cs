using Microsoft.ML;
using Microsoft.ML.Data;
using System.IO;
using System.Linq;


namespace MLDotNet_BaseballClassification
{
    public static class Utilities
    {
        /// <summary>
        /// Get Binary classfication metrics from a persisted model
        /// </summary>
        /// <param name="labelColumn"></param>
        /// <param name="algorithmTypeName"></param>
        /// <param name="validationData"></param>
        /// <returns></returns>
        public static CalibratedBinaryClassificationMetrics GetBinaryClassificationModelMetrics(bool isFinalModel, string appPath, MLContext mlContext, string labelColumn, string algorithmTypeName, IDataView validationData)
        {
            // Retrieve model path
            var loadedModelPath = Utilities.GetModelPath(appPath, algorithmTypeName, false, labelColumn, isFinalModel);

            // Load model for both prediction types
            var loadedModel = LoadModel(mlContext, loadedModelPath);

            // Apply the transformation pipeline to the data (i.e. normalization, probability score etc.)
            var transformedData = loadedModel.Transform(validationData);

            #if DEBUG
            var validationDataPreview = validationData.Preview(100);
            var transformedDataPreview = transformedData.Preview(100);
            #endif

            // Evaluate the model metrics using validation data
            var metrics = mlContext.BinaryClassification.Evaluate(transformedData, labelColumnName: labelColumn);

            return metrics;
        }

        /// <summary>
        /// Get Regression metrics for a persisted model
        /// </summary>
        /// <param name="appPath"></param>
        /// <param name="mlContext"></param>
        /// <param name="labelColumn"></param>
        /// <param name="algorithmTypeName"></param>
        /// <param name="validationData"></param>
        /// <returns></returns>
        public static RegressionMetrics GetRegressionModelMetrics(string appPath, MLContext mlContext, string labelColumn, string algorithmTypeName, IDataView validationData)
        {
            // Retrieve model path
            var loadedModelPath = Utilities.GetModelPath(appPath, algorithmTypeName, false, labelColumn, true);

            // Load model for both prediction types
            var loadedModel = LoadModel(mlContext, loadedModelPath);

            // Apply the transformation pipeline to the data (i.e. normalization, probability score etc.)
            var transformedData = loadedModel.Transform(validationData);

            #if DEBUG
            var validationDataPreview = validationData.Preview(100);
            var transformedDataPreview = transformedData.Preview(100);
            #endif

            // Evaluate the model metrics using validation data
            var metrics = mlContext.Regression.Evaluate(transformedData, labelColumnName: labelColumn);

            return metrics;
        }

        /// <summary>
        /// Get model path (ONNX, or MLNet (zip file)
        /// </summary>
        /// <param name="algorithmName"></param>
        /// <param name="isOnnx"></param>
        /// <param name="label"></param>
        /// <returns></returns>
        public static string GetModelPath(string appPath, string algorithmName, bool isOnnx, string label, bool isFinalModel)
        {
            // Model persistance convention used:
            // model + algorithmName + dependent variable column name + model persistance type extension (ONNX or native ML.NET)
            string modelPathName = string.Empty;
            string modelName = string.Format("{0}-{1}.onnx", label, algorithmName);
            string modelFolder = isFinalModel ? "Final" : "Test";

            if (isOnnx)
            {
                modelPathName = Path.Combine(appPath, "..", "..", "..", $@"Models\{modelFolder}", string.Format("{0}-{1}.onnx", label, algorithmName));
            }
            else
            {
                modelPathName = Path.Combine(appPath, "..", "..", "..", $@"Models\{modelFolder}", string.Format("{0}-{1}.mlnet", label, algorithmName));
            }

            return modelPathName;
        }

        /// <summary>
        /// Gets a baseline pipeline used for all of the models
        /// </summary>
        /// <returns></returns>
        public static EstimatorChain<Microsoft.ML.Transforms.NormalizingTransformer> GetBaseLinePipeline(MLContext mlContext, string[] featureColumns)
        {
            var baselineTransform = mlContext.Transforms.Concatenate("FeaturesBeforeNormalization", featureColumns)
                .Append(mlContext.Transforms.NormalizeMinMax("Features", "FeaturesBeforeNormalization"));


            return baselineTransform;
        }

        /// <summary>
        /// Load a ML.NET model from a persisted file location
        /// </summary>
        /// <param name="loadedModelPath"></param>
        /// <returns></returns>
        public static TransformerChain<ITransformer> LoadModel(MLContext mlContext, string loadedModelPath)
        {
            ITransformer loadedModel;
            DataViewSchema schema;

            using (var stream = File.OpenRead(loadedModelPath))
            {
                loadedModel = mlContext.Model.Load(stream, out schema);
            }

            // Cast the loaded model to a transformer chain
            // This allows for interacting with the LastTransformer property
            TransformerChain<ITransformer> transfomerChain = (TransformerChain<ITransformer>)loadedModel;

            return transfomerChain;
        }

        /// <summary>
        /// Save a model using default ML.NET persistance
        /// </summary>
        /// <param name="algorithmName"></param>
        /// <param name="labelColumn"></param>
        /// <param name="model"></param>
        public static void SaveModel(bool isFinalModel, string appPath, MLContext mlContext, DataViewSchema schema, string algorithmName, string labelColumn, ITransformer model)
        {
            var modelPath = GetModelPath(appPath, algorithmName, false, labelColumn, isFinalModel);

            // Write out the model
            using (var fileStream = new FileStream(modelPath, FileMode.Create, FileAccess.Write, FileShare.Write))
            {
                mlContext.Model.Save(model, schema, fileStream);
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
        public static void SaveOnnxModel(bool isFinalModel, string appPath, string algorithmName, string labelColumn, ITransformer model, MLContext mlContext, IDataView inputData)
        {
            var modelPath = GetModelPath(appPath, algorithmName, true, labelColumn, isFinalModel);

            if (SupportsOnnxPersistance(algorithmName))
            {
                // var protoBufModel = mlContext.Model.ConvertToOnnx(model, inputData, 

                // Persist the model (ONNX)
                using (var fileStream = new FileStream(modelPath, FileMode.Create, FileAccess.Write, FileShare.Write))
                {
                    mlContext.Model.ConvertToOnnx(model, inputData, fileStream);
                    //protoBufModel.WriteTo(fileStream);
                }
            }
        }

        /// <summary>
        /// Check if ML.NET algorithm supports ONNX persistance.
        /// </summary>
        /// <param name="algorithmName"></param>
        /// <returns></returns>
        public static bool SupportsOnnxPersistance(string algorithmName)
        {
            var algorithmsThatSupportOnnxPersistance = new string[]
                {"FastForest", "FastTree", "LightGbm", "LogisticRegression",
                "StochasticGradientDescentCalibrated"
               // , "StochasticGradientDescentNonCalibrated"
                };

            // Determine if algorithm is in the supported ONNX array
            var supportsOnnxPersitance = algorithmsThatSupportOnnxPersistance.Any(algorithmName.Contains);

            return supportsOnnxPersitance;
        }

        public static int GetMostContributingFeature(float[] featureContributions)
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
