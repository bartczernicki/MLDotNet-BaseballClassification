namespace MLDotNet_BaseballClassification.Services
{
    public class PlayerPredictionSample
    {
        public PlayerPredictionTier Tier { get; set; }
        public string DisplayName { get; set; } = string.Empty;
        public MLBBaseballBatter Batter { get; set; } = new MLBBaseballBatter();
    }
}
