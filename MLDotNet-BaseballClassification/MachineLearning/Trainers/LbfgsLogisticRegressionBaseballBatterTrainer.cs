using Microsoft.ML;
using Microsoft.ML.Calibrators;
using Microsoft.ML.Trainers;

namespace MLDotNet_BaseballClassification.MachineLearning.Trainers
{
    public class LbfgsLogisticRegressionBaseballBatterTrainer : BaseballBatterTrainerBase<
                  CalibratedModelParametersBase<LinearBinaryModelParameters, PlattCalibrator>>
    {
        public LbfgsLogisticRegressionBaseballBatterTrainer(string labelColumnName)
    : base()
        {
            this.AlgorithmName = "LbfgsLogisticRegression";
            this.Name = $"LbfgsLogisticRegression-{labelColumnName}|";
            this.LabelColumnName = labelColumnName;

            _trainerEstimator = _mlContext.BinaryClassification.Trainers.LbfgsLogisticRegression(labelColumnName: labelColumnName);
        }
    }
}
