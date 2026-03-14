using System.Text.Json;

namespace SDSharp.Onnx
{
    internal sealed class StableDiffusionScheduler
    {
        private readonly float[] AlphasCumProd;
        private readonly float FinalAlphaCumProd;

        public int NumTrainTimesteps { get; }
        public int StepsOffset { get; }
        public bool ClipSample { get; }
        public string PredictionType { get; }



        private StableDiffusionScheduler(float[] betas, int numTrainTimesteps, int stepsOffset, bool clipSample, bool setAlphaToOne, string predictionType)
        {
            this.NumTrainTimesteps = numTrainTimesteps;
            this.StepsOffset = stepsOffset;
            this.ClipSample = clipSample;
            this.PredictionType = string.IsNullOrWhiteSpace(predictionType) ? "epsilon" : predictionType;

            this.AlphasCumProd = new float[betas.Length];
            float cumulative = 1.0f;
            for (int i = 0; i < betas.Length; i++)
            {
                cumulative *= 1.0f - betas[i];
                this.AlphasCumProd[i] = cumulative;
            }

            this.FinalAlphaCumProd = setAlphaToOne ? 1.0f : this.AlphasCumProd[0];
        }



        public static StableDiffusionScheduler Create(JsonElement root)
        {
            int numTrainTimesteps = TryGetInt(root, "num_train_timesteps", 1000);
            int stepsOffset = TryGetInt(root, "steps_offset", 1);
            bool clipSample = TryGetBool(root, "clip_sample", false);
            bool setAlphaToOne = TryGetBool(root, "set_alpha_to_one", false);
            string predictionType = TryGetString(root, "prediction_type", "epsilon");

            float[] betas = TryGetFloatArray(root, "trained_betas")
                ?? BuildBetas(
                    numTrainTimesteps,
                    TryGetFloat(root, "beta_start", 0.00085f),
                    TryGetFloat(root, "beta_end", 0.012f),
                    TryGetString(root, "beta_schedule", "scaled_linear"));

            return new StableDiffusionScheduler(betas, numTrainTimesteps, stepsOffset, clipSample, setAlphaToOne, predictionType);
        }



        public int[] CreateTimesteps(int inferenceSteps)
        {
            int stepRatio = Math.Max(1, this.NumTrainTimesteps / inferenceSteps);
            var timesteps = new int[inferenceSteps];

            for (int i = 0; i < inferenceSteps; i++)
            {
                timesteps[i] = Math.Min(((inferenceSteps - 1 - i) * stepRatio) + this.StepsOffset, this.NumTrainTimesteps - 1);
            }

            return timesteps;
        }



        public float[] Step(ReadOnlySpan<float> noisePrediction, ReadOnlySpan<float> sample, int timestep, int prevTimestep)
        {
            float alphaProdT = this.AlphasCumProd[timestep];
            float alphaProdPrev = prevTimestep >= 0 ? this.AlphasCumProd[prevTimestep] : this.FinalAlphaCumProd;
            float betaProdT = 1.0f - alphaProdT;
            float betaProdPrev = 1.0f - alphaProdPrev;

            float sqrtAlphaProdT = MathF.Sqrt(alphaProdT);
            float sqrtBetaProdT = MathF.Sqrt(betaProdT);
            float sqrtAlphaProdPrev = MathF.Sqrt(alphaProdPrev);
            float sqrtBetaProdPrev = MathF.Sqrt(betaProdPrev);

            var previousSample = new float[sample.Length];

            for (int i = 0; i < sample.Length; i++)
            {
                float predictedNoise = noisePrediction[i];
                float predictedOriginalSample;

                if (string.Equals(this.PredictionType, "sample", StringComparison.OrdinalIgnoreCase))
                {
                    predictedOriginalSample = predictedNoise;
                    predictedNoise = (sample[i] - (sqrtAlphaProdT * predictedOriginalSample)) / MathF.Max(sqrtBetaProdT, 1e-6f);
                }
                else if (string.Equals(this.PredictionType, "v_prediction", StringComparison.OrdinalIgnoreCase))
                {
                    predictedOriginalSample = (sqrtAlphaProdT * sample[i]) - (sqrtBetaProdT * predictedNoise);
                    predictedNoise = (sqrtAlphaProdT * predictedNoise) + (sqrtBetaProdT * sample[i]);
                }
                else
                {
                    predictedOriginalSample = (sample[i] - (sqrtBetaProdT * predictedNoise)) / MathF.Max(sqrtAlphaProdT, 1e-6f);
                }

                if (this.ClipSample)
                {
                    predictedOriginalSample = Math.Clamp(predictedOriginalSample, -1.0f, 1.0f);
                }

                previousSample[i] = (sqrtAlphaProdPrev * predictedOriginalSample) + (sqrtBetaProdPrev * predictedNoise);
            }

            return previousSample;
        }



        private static float[] BuildBetas(int count, float betaStart, float betaEnd, string schedule)
        {
            var betas = new float[count];

            if (string.Equals(schedule, "scaled_linear", StringComparison.OrdinalIgnoreCase))
            {
                float start = MathF.Sqrt(betaStart);
                float end = MathF.Sqrt(betaEnd);
                for (int i = 0; i < count; i++)
                {
                    float value = Lerp(start, end, i / (float) Math.Max(1, count - 1));
                    betas[i] = value * value;
                }

                return betas;
            }

            for (int i = 0; i < count; i++)
            {
                betas[i] = Lerp(betaStart, betaEnd, i / (float) Math.Max(1, count - 1));
            }

            return betas;
        }



        private static float Lerp(float start, float end, float amount)
        {
            return start + ((end - start) * amount);
        }



        private static float?[]? TryGetNullableFloatArray(JsonElement root, string propertyName)
        {
            if (!root.TryGetProperty(propertyName, out JsonElement property) || property.ValueKind == JsonValueKind.Null)
            {
                return null;
            }

            if (property.ValueKind != JsonValueKind.Array)
            {
                return null;
            }

            return property.EnumerateArray().Select(static item => item.ValueKind == JsonValueKind.Null ? (float?) null : item.GetSingle()).ToArray();
        }



        private static float[]? TryGetFloatArray(JsonElement root, string propertyName)
        {
            float?[]? nullable = TryGetNullableFloatArray(root, propertyName);
            return nullable?.Where(static value => value.HasValue).Select(static value => value!.Value).ToArray();
        }



        private static float TryGetFloat(JsonElement root, string propertyName, float defaultValue)
        {
            if (root.TryGetProperty(propertyName, out JsonElement property) && property.ValueKind == JsonValueKind.Number)
            {
                return property.GetSingle();
            }

            return defaultValue;
        }



        private static int TryGetInt(JsonElement root, string propertyName, int defaultValue)
        {
            if (root.TryGetProperty(propertyName, out JsonElement property) && property.ValueKind == JsonValueKind.Number)
            {
                return property.GetInt32();
            }

            return defaultValue;
        }



        private static bool TryGetBool(JsonElement root, string propertyName, bool defaultValue)
        {
            if (root.TryGetProperty(propertyName, out JsonElement property) && (property.ValueKind == JsonValueKind.True || property.ValueKind == JsonValueKind.False))
            {
                return property.GetBoolean();
            }

            return defaultValue;
        }



        private static string TryGetString(JsonElement root, string propertyName, string defaultValue)
        {
            if (root.TryGetProperty(propertyName, out JsonElement property) && property.ValueKind == JsonValueKind.String)
            {
                return property.GetString() ?? defaultValue;
            }

            return defaultValue;
        }
    }
}
