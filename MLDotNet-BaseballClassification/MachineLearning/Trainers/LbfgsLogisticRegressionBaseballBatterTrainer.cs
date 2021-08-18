using Microsoft.ML;
using Microsoft.ML.Calibrators;
using Microsoft.ML.Trainers;

namespace MLDotNet_BaseballClassification.MachineLearning.Trainers
{
    public class LbfgsLogisticRegressionBaseballBatterTrainer : BaseballBatterTrainerBase<
                  CalibratedModelParametersBase<LinearBinaryModelParameters, PlattCalibrator>>
    {
        public LbfgsLogisticRegressionBaseballBatterTrainer(string labelColumnName, float l1Regularization = 1f,
            float l2Regularization = 1f, float optimizationTolerance = 0.0000001f)
    : base()
        {
            this.AlgorithmName = "LogisticRegression";
            this.Name = $"LogisticRegression-{labelColumnName}|{l1Regularization}|{l2Regularization}|{optimizationTolerance}";
            this.LabelColumnName = labelColumnName;

            _trainerEstimator = _mlContext.BinaryClassification.Trainers.LbfgsLogisticRegression(labelColumnName: labelColumnName,
                l1Regularization: l1Regularization, l2Regularization: l2Regularization, optimizationTolerance: optimizationTolerance);
        }
    }
}
