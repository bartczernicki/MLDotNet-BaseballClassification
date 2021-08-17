using Microsoft.ML;
using Microsoft.ML.Calibrators;
using Microsoft.ML.Trainers;

namespace MLDotNet_BaseballClassification.MachineLearning.Trainers
{
    public class SgdCalibratedBaseballBatterTrainer : BaseballBatterTrainerBase<
                  CalibratedModelParametersBase<LinearBinaryModelParameters, PlattCalibrator>>
    {
        public SgdCalibratedBaseballBatterTrainer(string labelColumnName, int numberOfIterations = 10)
    : base()
        {
            this.AlgorithmName = "SgdCalibrated";
            this.Name = $"SgdCalibrated-{labelColumnName}|{numberOfIterations}";
            this.LabelColumnName = labelColumnName;

            _trainerEstimator = _mlContext.BinaryClassification.Trainers.SgdCalibrated(labelColumnName: labelColumnName, 
                numberOfIterations: numberOfIterations);
        }
    }
}
