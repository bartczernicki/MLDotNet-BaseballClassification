using Microsoft.ML;
using Microsoft.ML.Calibrators;
using Microsoft.ML.Trainers;

namespace MLDotNet_BaseballClassification.MachineLearning.Trainers
{
    public class SgdCalibratedBaseballBatterTrainer : BaseballBatterTrainerBase<
                  CalibratedModelParametersBase<LinearBinaryModelParameters, PlattCalibrator>>
    {
        public SgdCalibratedBaseballBatterTrainer(string labelColumnName, int numberOfIterations = 20,
            double learningRate = 0.01, float l2Regularization = 0.000001f)
    : base()
        {
            this.AlgorithmName = "SgdCalibrated";
            this.Name = $"SgdCalibrated-{labelColumnName}|{numberOfIterations}|{learningRate}|{l2Regularization}";
            this.LabelColumnName = labelColumnName;

            _trainerEstimator = _mlContext.BinaryClassification.Trainers.SgdCalibrated(labelColumnName: labelColumnName, 
                numberOfIterations: numberOfIterations, learningRate: learningRate, l2Regularization: l2Regularization);
        }
    }
}
