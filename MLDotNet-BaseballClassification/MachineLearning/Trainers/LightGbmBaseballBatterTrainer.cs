using Microsoft.ML;
using Microsoft.ML.Calibrators;
using Microsoft.ML.Trainers.LightGbm;

namespace MLDotNet_BaseballClassification.MachineLearning.Trainers
{
    public class LightGbmBaseballBatterTrainer : BaseballBatterTrainerBase<
                  CalibratedModelParametersBase<LightGbmBinaryModelParameters, PlattCalibrator>>
    {
        public LightGbmBaseballBatterTrainer(string labelColumnName, int? numberOfLeaves = null,
            int? minimumExampleCountPerLeaf = null, double? learningRate = null,
            int numberOfIterations = 100)
    : base()
        {
            this.AlgorithmName = "LightGbm";
            this.Name = $"LightGbm-{labelColumnName}|{numberOfIterations}";
            this.LabelColumnName = labelColumnName;

            _trainerEstimator = _mlContext.BinaryClassification.Trainers.LightGbm(labelColumnName: labelColumnName,
                numberOfLeaves: numberOfLeaves, minimumExampleCountPerLeaf: minimumExampleCountPerLeaf,
                learningRate: learningRate, numberOfIterations: numberOfIterations);
        }
    }
}
