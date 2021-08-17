using Microsoft.ML;
using Microsoft.ML.Calibrators;
using Microsoft.ML.Trainers.FastTree;

namespace MLDotNet_BaseballClassification.MachineLearning.Trainers
{
    public class FastTreeBaseballBatterTrainer : BaseballBatterTrainerBase<
                  CalibratedModelParametersBase<FastTreeBinaryModelParameters, PlattCalibrator>>
    {
        public FastTreeBaseballBatterTrainer(string labelColumnName, int numberOfLeaves = 20, int numberOfTrees = 100, 
            int minimumExampleCountPerLeaf = 10, double learningRate = 0.2)
    : base()
        {
            this.AlgorithmName = "FastTree";
            this.Name = $"Fast Tree-{labelColumnName}|{numberOfLeaves}-{numberOfTrees}-{minimumExampleCountPerLeaf}-{learningRate}";
            this.LabelColumnName = labelColumnName;

            _trainerEstimator = _mlContext.BinaryClassification.Trainers.FastTree(labelColumnName: labelColumnName,
                numberOfLeaves: numberOfLeaves,numberOfTrees: numberOfTrees, minimumExampleCountPerLeaf: minimumExampleCountPerLeaf,
                                      learningRate: learningRate);
        }
    }
}
