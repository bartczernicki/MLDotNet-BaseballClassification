using Microsoft.ML;
using Microsoft.ML.Calibrators;
using Microsoft.ML.Trainers;
using Microsoft.ML.Trainers.FastTree;

namespace MLDotNet_BaseballClassification.MachineLearning.Trainers
{
    public class AveragePerceptronBaseballBatterTrainer : BaseballBatterTrainerBase<LinearBinaryModelParameters>
    {
        public AveragePerceptronBaseballBatterTrainer(string labelColumnName, int numberOfIterations = 10)
    : base()
        {
            this.AlgorithmName = "LightGbm";
            this.Name = $"LightGbm-{labelColumnName}|{numberOfIterations}";
            this.LabelColumnName = labelColumnName;

            _trainerEstimator = _mlContext.BinaryClassification.Trainers.AveragedPerceptron(labelColumnName: labelColumnName,
                numberOfIterations: numberOfIterations);
        }
    }
}
