using System;
using System.Collections.Generic;

namespace MLDotNet_BaseballClassification.Services
{
    public class FictitiousPlayerSampleService
    {
        public IReadOnlyList<PlayerPredictionSample> GetSamples()
        {
            return new List<PlayerPredictionSample>
            {
                // Bad player profiles (10)
                CreateSample(PlayerPredictionTier.Bad, "BAD001", "Bad Player 01", 2f, 250f, 20f, 50f, 8f, 1f, 1f, 18f, 3f, 0.200f, 0.252f, 0f, 63f, 0f),
                CreateSample(PlayerPredictionTier.Bad, "BAD002", "Bad Player 02", 3f, 480f, 35f, 95f, 12f, 2f, 4f, 40f, 8f, 0.198f, 0.256f, 0f, 123f, 0f),
                CreateSample(PlayerPredictionTier.Bad, "BAD003", "Bad Player 03", 4f, 620f, 44f, 132f, 18f, 2f, 6f, 52f, 5f, 0.213f, 0.277f, 0f, 172f, 0f),
                CreateSample(PlayerPredictionTier.Bad, "BAD004", "Bad Player 04", 5f, 850f, 71f, 190f, 24f, 4f, 10f, 78f, 11f, 0.224f, 0.296f, 0f, 252f, 0f),
                CreateSample(PlayerPredictionTier.Bad, "BAD005", "Bad Player 05", 6f, 1100f, 88f, 250f, 30f, 4f, 14f, 95f, 14f, 0.227f, 0.300f, 0f, 330f, 0f),
                CreateSample(PlayerPredictionTier.Bad, "BAD006", "Bad Player 06", 7f, 1350f, 102f, 289f, 36f, 5f, 18f, 116f, 16f, 0.214f, 0.288f, 0f, 389f, 0f),
                CreateSample(PlayerPredictionTier.Bad, "BAD007", "Bad Player 07", 8f, 1700f, 130f, 360f, 42f, 6f, 22f, 145f, 21f, 0.212f, 0.282f, 1f, 480f, 1f),
                CreateSample(PlayerPredictionTier.Bad, "BAD008", "Bad Player 08", 9f, 2100f, 171f, 470f, 55f, 7f, 28f, 190f, 24f, 0.224f, 0.297f, 1f, 623f, 1f),
                CreateSample(PlayerPredictionTier.Bad, "BAD009", "Bad Player 09", 10f, 2600f, 205f, 585f, 68f, 8f, 32f, 225f, 28f, 0.225f, 0.294f, 1f, 765f, 1f),
                CreateSample(PlayerPredictionTier.Bad, "BAD010", "Bad Player 10", 11f, 3200f, 245f, 700f, 80f, 9f, 38f, 275f, 32f, 0.219f, 0.285f, 1f, 912f, 1f),

                // Average player profiles (10)
                CreateSample(PlayerPredictionTier.Average, "AVG001", "Average Player 01", 6f, 1400f, 180f, 360f, 62f, 10f, 20f, 165f, 22f, 0.257f, 0.359f, 1f, 502f, 1f),
                CreateSample(PlayerPredictionTier.Average, "AVG002", "Average Player 02", 7f, 1750f, 240f, 455f, 78f, 12f, 27f, 210f, 30f, 0.260f, 0.365f, 1f, 638f, 1f),
                CreateSample(PlayerPredictionTier.Average, "AVG003", "Average Player 03", 8f, 2100f, 280f, 560f, 95f, 14f, 35f, 270f, 40f, 0.267f, 0.375f, 2f, 788f, 2f),
                CreateSample(PlayerPredictionTier.Average, "AVG004", "Average Player 04", 9f, 2500f, 320f, 655f, 110f, 15f, 42f, 320f, 48f, 0.262f, 0.368f, 2f, 921f, 2f),
                CreateSample(PlayerPredictionTier.Average, "AVG005", "Average Player 05", 10f, 2900f, 370f, 760f, 128f, 16f, 52f, 390f, 55f, 0.262f, 0.371f, 3f, 1076f, 3f),
                CreateSample(PlayerPredictionTier.Average, "AVG006", "Average Player 06", 11f, 3400f, 430f, 910f, 150f, 18f, 68f, 470f, 62f, 0.268f, 0.382f, 3f, 1300f, 4f),
                CreateSample(PlayerPredictionTier.Average, "AVG007", "Average Player 07", 12f, 3900f, 500f, 1040f, 172f, 20f, 82f, 560f, 70f, 0.267f, 0.384f, 4f, 1498f, 5f),
                CreateSample(PlayerPredictionTier.Average, "AVG008", "Average Player 08", 13f, 4500f, 590f, 1220f, 198f, 22f, 105f, 670f, 78f, 0.271f, 0.395f, 5f, 1777f, 6f),
                CreateSample(PlayerPredictionTier.Average, "AVG009", "Average Player 09", 14f, 5200f, 700f, 1410f, 225f, 24f, 132f, 790f, 86f, 0.271f, 0.400f, 6f, 2079f, 8f),
                CreateSample(PlayerPredictionTier.Average, "AVG010", "Average Player 10", 15f, 6000f, 810f, 1650f, 255f, 26f, 165f, 930f, 95f, 0.275f, 0.409f, 7f, 2452f, 10f),

                // Great player profiles (10)
                CreateSample(PlayerPredictionTier.Great, "GRT001", "Great Player 01", 14f, 6200f, 850f, 2050f, 350f, 40f, 320f, 1250f, 130f, 0.331f, 0.555f, 10f, 3440f, 18f),
                CreateSample(PlayerPredictionTier.Great, "GRT002", "Great Player 02", 15f, 7000f, 940f, 2200f, 380f, 45f, 360f, 1380f, 140f, 0.314f, 0.536f, 11f, 3750f, 22f),
                CreateSample(PlayerPredictionTier.Great, "GRT003", "Great Player 03", 16f, 7600f, 1010f, 2350f, 410f, 50f, 390f, 1500f, 150f, 0.309f, 0.530f, 12f, 4030f, 26f),
                CreateSample(PlayerPredictionTier.Great, "GRT004", "Great Player 04", 17f, 8200f, 1100f, 2550f, 430f, 55f, 430f, 1650f, 170f, 0.311f, 0.534f, 13f, 4380f, 30f),
                CreateSample(PlayerPredictionTier.Great, "GRT005", "Great Player 05", 18f, 9000f, 1200f, 2800f, 460f, 60f, 480f, 1820f, 190f, 0.311f, 0.536f, 14f, 4820f, 34f),
                CreateSample(PlayerPredictionTier.Great, "GRT006", "Great Player 06", 19f, 9800f, 1300f, 3050f, 490f, 65f, 530f, 2000f, 210f, 0.311f, 0.537f, 15f, 5260f, 39f),
                CreateSample(PlayerPredictionTier.Great, "GRT007", "Great Player 07", 20f, 10600f, 1390f, 3320f, 520f, 70f, 575f, 2160f, 235f, 0.313f, 0.538f, 16f, 5705f, 44f),
                CreateSample(PlayerPredictionTier.Great, "GRT008", "Great Player 08", 21f, 11400f, 1490f, 3560f, 550f, 75f, 620f, 2320f, 260f, 0.312f, 0.537f, 17f, 6120f, 49f),
                CreateSample(PlayerPredictionTier.Great, "GRT009", "Great Player 09", 22f, 12300f, 1580f, 3800f, 580f, 80f, 665f, 2460f, 285f, 0.309f, 0.531f, 18f, 6535f, 54f),
                CreateSample(PlayerPredictionTier.Great, "GRT010", "Great Player 10", 23f, 13200f, 1660f, 4050f, 610f, 85f, 710f, 2620f, 305f, 0.307f, 0.527f, 19f, 6960f, 60f)
            };
        }

        private static PlayerPredictionSample CreateSample(
            PlayerPredictionTier tier,
            string id,
            string displayName,
            float yearsPlayed,
            float ab,
            float runs,
            float hits,
            float doubles,
            float triples,
            float homeRuns,
            float runsBattedIn,
            float stolenBases,
            float battingAverage,
            float sluggingPct,
            float allStarAppearances,
            float totalBases,
            float totalPlayerAwards)
        {
            // Keep the sample catalog internally consistent with baseball rate stats so
            // slash lines still make sense when someone tweaks the counting stats later.
            ValidateSampleMetrics(
                id,
                yearsPlayed,
                ab,
                hits,
                doubles,
                triples,
                homeRuns,
                runsBattedIn,
                battingAverage,
                sluggingPct,
                allStarAppearances,
                totalBases);

            return new PlayerPredictionSample
            {
                Tier = tier,
                DisplayName = displayName,
                Batter = new MLBBaseballBatter
                {
                    FullPlayerName = displayName,
                    ID = id,
                    InductedToHallOfFame = false,
                    OnHallOfFameBallot = false,
                    YearsPlayed = yearsPlayed,
                    AB = ab,
                    R = runs,
                    H = hits,
                    Doubles = doubles,
                    Triples = triples,
                    HR = homeRuns,
                    RBI = runsBattedIn,
                    SB = stolenBases,
                    BattingAverage = battingAverage,
                    SluggingPct = sluggingPct,
                    AllStarAppearances = allStarAppearances,
                    TB = totalBases,
                    TotalPlayerAwards = totalPlayerAwards,
                    LastYearPlayed = 0f
                }
            };
        }

        private static void ValidateSampleMetrics(
            string id,
            float yearsPlayed,
            float atBats,
            float hits,
            float doubles,
            float triples,
            float homeRuns,
            float runsBattedIn,
            float battingAverage,
            float sluggingPct,
            float allStarAppearances,
            float totalBases)
        {
            if (atBats <= 0f)
            {
                throw new InvalidOperationException($"Sample {id} must have positive at-bats.");
            }

            if (hits < 0f || hits > atBats)
            {
                throw new InvalidOperationException($"Sample {id} has impossible hit totals for its at-bats.");
            }

            if (allStarAppearances > yearsPlayed)
            {
                throw new InvalidOperationException($"Sample {id} cannot have more All-Star appearances than seasons played.");
            }

            var singles = hits - doubles - triples - homeRuns;
            if (singles < 0f)
            {
                throw new InvalidOperationException($"Sample {id} has more extra-base hits than total hits.");
            }

            if (runsBattedIn < homeRuns)
            {
                throw new InvalidOperationException($"Sample {id} cannot have fewer RBI than home runs.");
            }

            var calculatedTotalBases = singles + (2f * doubles) + (3f * triples) + (4f * homeRuns);
            if (Math.Abs(calculatedTotalBases - totalBases) > 0.001f)
            {
                throw new InvalidOperationException($"Sample {id} has total bases that do not match its hit breakdown.");
            }

            var expectedBattingAverage = RoundRateStat(hits / atBats);
            if (Math.Abs(expectedBattingAverage - battingAverage) > 0.0005f)
            {
                throw new InvalidOperationException($"Sample {id} has a batting average that does not match hits divided by at-bats.");
            }

            var expectedSluggingPct = RoundRateStat(totalBases / atBats);
            if (Math.Abs(expectedSluggingPct - sluggingPct) > 0.0005f)
            {
                throw new InvalidOperationException($"Sample {id} has a slugging percentage that does not match total bases divided by at-bats.");
            }
        }

        private static float RoundRateStat(float value)
        {
            return (float)Math.Round(value, 3, MidpointRounding.AwayFromZero);
        }
    }
}
