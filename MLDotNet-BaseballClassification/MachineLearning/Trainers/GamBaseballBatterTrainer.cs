using Microsoft.ML;
using Microsoft.ML.Calibrators;
using Microsoft.ML.Trainers.FastTree;

namespace MLDotNet_BaseballClassification.MachineLearning.Trainers
{
    public class GamBaseballBatterTrainer : BaseballBatterTrainerBase<
                  CalibratedModelParametersBase<GamBinaryModelParameters, PlattCalibrator>>
    {
        public GamBaseballBatterTrainer(string labelColumnName, int numberOfIterations = 9500, int maximumBinCountPerFeature = 255,
            double learningRate = 0.002)
    : base()
        {
            this.AlgorithmName = "GeneralizedAdditiveModels";
            this.Name = $"GeneralizedAdditiveModels-{labelColumnName}|{numberOfIterations}-{maximumBinCountPerFeature}-{learningRate}";
            this.LabelColumnName = labelColumnName;

            _trainerEstimator = _mlContext.BinaryClassification.Trainers.Gam(labelColumnName: labelColumnName,
                numberOfIterations: numberOfIterations, maximumBinCountPerFeature: maximumBinCountPerFeature,
                learningRate: learningRate);
        }
    }
}
