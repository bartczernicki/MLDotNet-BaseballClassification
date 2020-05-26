using Microsoft.ML.Data;
using System;

namespace MLDotNet_BaseballClassification
{
    public class MLBBaseballBatterSimple
    {
        [LoadColumn(0), ColumnName("InductedToHallOfFame")]
        public bool InductedToHallOfFame { get; set; }

        [LoadColumn(3), ColumnName("YearsPlayed")]
        public float YearsPlayed { get; set; }
    }
}
