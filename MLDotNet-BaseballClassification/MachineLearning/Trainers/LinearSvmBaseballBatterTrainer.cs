using Microsoft.ML;
using Microsoft.ML.Trainers;

namespace MLDotNet_BaseballClassification.MachineLearning.Trainers
{
    public class LinearSvmBaseballBatterTrainer : BaseballBatterTrainerBase<LinearBinaryModelParameters>
    {
        public LinearSvmBaseballBatterTrainer(string labelColumnName, int numberOfIterations = 1)
    : base()
        {
            this.AlgorithmName = "LinearSvm";
            this.Name = $"LinearSvm-{labelColumnName}|{numberOfIterations}";
            this.LabelColumnName = labelColumnName;

            _trainerEstimator = _mlContext.BinaryClassification.Trainers.LinearSvm(labelColumnName: labelColumnName,
                numberOfIterations: numberOfIterations);
        }
    }
}
