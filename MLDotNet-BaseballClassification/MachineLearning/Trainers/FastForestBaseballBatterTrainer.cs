using Microsoft.ML;
using Microsoft.ML.Calibrators;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers.FastTree;

namespace MLDotNet_BaseballClassification.MachineLearning.Trainers
{
    public class FastForestBaseballBatterTrainer : BaseballBatterTrainerBase<FastForestBinaryModelParameters>
    {
        public FastForestBaseballBatterTrainer(string labelColumnName, int numberOfLeaves = 20, int numberOfTrees = 100,
            int minimumExampleCountPerLeaf = 10)
    : base()
        {
            this.AlgorithmName = "FastForest";
            this.Name = $"Fast Forest-{labelColumnName}|{numberOfLeaves}-{numberOfTrees}-{minimumExampleCountPerLeaf}";
            this.LabelColumnName = labelColumnName;

            _trainerEstimator = _mlContext.BinaryClassification.Trainers.FastForest(labelColumnName: labelColumnName,
                numberOfLeaves: numberOfLeaves, numberOfTrees: numberOfTrees, minimumExampleCountPerLeaf: minimumExampleCountPerLeaf);
        }
    }
}
