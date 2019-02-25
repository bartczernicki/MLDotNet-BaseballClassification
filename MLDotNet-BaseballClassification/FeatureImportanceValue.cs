using System;
using System.Collections.Generic;
using System.Text;

namespace MLDotNet_BaseballClassification
{
    public class FeatureImportanceValue
    {
        public string FeatureName { get; set; }
        public string PerformanceMetricName { get; set; }
        public double  PerformanceMetricValue { get; set; }
    }
}
