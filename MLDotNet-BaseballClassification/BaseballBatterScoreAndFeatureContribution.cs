using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Text;

namespace MLDotNet_BaseballClassification
{
    public class BaseballBatterScoreAndFeatureContribution
    {
        public float Probability { get; set; }

        [VectorType(11)]
        public float[] Features { get; set; }

        public float Score { get; set; }

        [VectorType(4)]
        public float[] FeatureContributions { get; set; }
    }
}
