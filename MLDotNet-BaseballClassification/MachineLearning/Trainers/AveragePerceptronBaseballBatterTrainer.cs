using Microsoft.ML;
using Microsoft.ML.Trainers;

namespace MLDotNet_BaseballClassification.MachineLearning.Trainers
{
    public class AveragePerceptronBaseballBatterTrainer : BaseballBatterTrainerBase<LinearBinaryModelParameters>
    {
        public AveragePerceptronBaseballBatterTrainer(string labelColumnName, int numberOfIterations = 10)
    : base()
        {
            this.AlgorithmName = "AveragePerceptron";
            this.Name = $"AveragePerceptron-{labelColumnName}|{numberOfIterations}";
            this.LabelColumnName = labelColumnName;

            _trainerEstimator = _mlContext.BinaryClassification.Trainers.AveragedPerceptron(labelColumnName: labelColumnName,
                numberOfIterations: numberOfIterations);
        }
    }
}
