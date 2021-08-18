using Microsoft.ML;
using Microsoft.ML.Trainers;

namespace MLDotNet_BaseballClassification.MachineLearning.Trainers
{
    public class AveragedPerceptronBaseballBatterTrainer : BaseballBatterTrainerBase<LinearBinaryModelParameters>
    {
        public AveragedPerceptronBaseballBatterTrainer(string labelColumnName, int numberOfIterations = 10)
    : base()
        {
            this.AlgorithmName = "AveragedPerceptron";
            this.Name = $"AveragedPerceptron-{labelColumnName}|{numberOfIterations}";
            this.LabelColumnName = labelColumnName;

            _trainerEstimator = _mlContext.BinaryClassification.Trainers.AveragedPerceptron(labelColumnName: labelColumnName,
                numberOfIterations: numberOfIterations);
        }
    }
}
