using Microsoft.Data.DataView;
using Microsoft.ML;
using System;
using System.Collections.Generic;
using System.Text;

namespace MLDotNet_BaseballClassification
{
    public class BaseballModel
    {
        public string LabelColumn { get; set; }
        public MLContext MLNetContext { get; set; }
        public IDataView TrainingData { get; set; }
        public string BinaryClassificationAlgorithm { get; set; }

        public BaseballModel(string labelColumn, MLContext mlNetContext, IDataView trainingData, string binaryClassificationAlgorithm)
        {
            this.LabelColumn = labelColumn;
            this.MLNetContext = mlNetContext;
            this.TrainingData = trainingData;
            this.BinaryClassificationAlgorithm = binaryClassificationAlgorithm;
        }

        //public 
    }
}
