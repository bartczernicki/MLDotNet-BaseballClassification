using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace MLDotNet_BaseballClassification.MachineLearning
{
    public interface ITrainerBase
    {
        string AlgorithmName { get; }

        string LabelColumnName { get; }

        string Name { get; }

        void Fit(IDataView trainingData);

        BinaryClassificationMetrics Evaluate(IDataView testData);

        void SaveModel(string folderPath, bool isOnnx, bool isFinalModel);
    }
}
