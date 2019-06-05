using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Text;

namespace MLDotNet_BaseballClassification
{
    public class BaseballClassificationModel
    {
        public string LabelColumn { get; set; }
        public string BinaryClassificationAlgorithm { get; set; }
        public BinaryClassificationMetricsStatistics Metrics { get; set; }

        public BaseballClassificationModel(string labelColumn, string binaryClassificationAlgorithm, BinaryClassificationMetricsStatistics metrics)
        {
            this.LabelColumn = labelColumn;
            this.BinaryClassificationAlgorithm = binaryClassificationAlgorithm;
            this.Metrics = Metrics;
        }

        //public 
    }
}
