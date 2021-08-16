using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML.Data;

namespace MLDotNet_BaseballClassification.MachineLearning
{
    public interface ITrainerBase
    {
        string Name { get; }
        void Fit(string trainingFileName);
        BinaryClassificationMetrics Evaluate();
        void SaveModel();
    }
}
