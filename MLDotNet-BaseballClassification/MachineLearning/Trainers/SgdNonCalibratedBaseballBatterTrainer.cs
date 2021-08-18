using Microsoft.ML;
using Microsoft.ML.Trainers;

namespace MLDotNet_BaseballClassification.MachineLearning.Trainers
{
    public class SgdNonCalibratedBaseballBatterTrainer : BaseballBatterTrainerBase<LinearBinaryModelParameters>
    {
        public SgdNonCalibratedBaseballBatterTrainer(string labelColumnName, int numberOfIterations = 20,
            double learningRate = 0.01, float l2Regularization = 0.000001f)
    : base()
        {
            this.AlgorithmName = "StochasticGradientDescentNonCalibrated";
            this.Name = $"StochasticGradientDescentNonCalibrated-{labelColumnName}|{numberOfIterations}|{learningRate}|{l2Regularization}";
            this.LabelColumnName = labelColumnName;

            _trainerEstimator = _mlContext.BinaryClassification.Trainers.SgdNonCalibrated(labelColumnName: labelColumnName,
                numberOfIterations: numberOfIterations, learningRate: learningRate, l2Regularization: l2Regularization);
        }
    }
}