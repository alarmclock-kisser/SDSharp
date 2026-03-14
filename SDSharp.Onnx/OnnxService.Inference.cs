using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SDSharp.Shared;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Formats.Png;
using SixLabors.ImageSharp.PixelFormats;
using System.Runtime.Versioning;

namespace SDSharp.Onnx
{
    public partial class OnnxService
    {
        [SupportedOSPlatform("windows")]
        public async Task<ImageObj?> GenerateImageAsync(StableDiffusionGenerateRequest request, IProgress<double>? progress = null, CancellationToken ct = default)
        {
            request ??= new StableDiffusionGenerateRequest();

            var validationErrors = request.Validate();
            if (validationErrors.Count > 0)
            {
                foreach (var error in validationErrors.Distinct())
                {
                    await StaticLogger.LogAsync($"Generate validation failed: {error}");
                }

                return null;
            }

            await this.StateLock.WaitAsync(ct);
            try
            {
                progress?.Report(0.0);

                StableDiffusionModel? activeModel = this.LoadedModel;
                if (request.Model != null && (activeModel == null || !activeModel.IsSameModelAs(request.Model)))
                {
                    if (!request.AutoLoadModel)
                    {
                        await StaticLogger.LogAsync("No matching model is loaded and AutoLoadModel is false.");
                        return null;
                    }

                    activeModel = await this.LoadModelCoreAsync(
                        request.Model,
                        request.LoadOptions ?? this.LoadedModelOptions ?? new StableDiffusionLoadOptions(),
                        this.CreateNestedProgress(progress, 0.0, 0.25),
                        ct);
                }
                else if (activeModel == null)
                {
                    await StaticLogger.LogAsync("No model is currently loaded.");
                    return null;
                }

                if (activeModel == null)
                {
                    return null;
                }

                return await this.GenerateWithRecoveryAsync(request, activeModel, progress, ct);
            }
            finally
            {
                this.StateLock.Release();
            }
        }

        [SupportedOSPlatform("windows")]
        private async Task<ImageObj?> GenerateWithRecoveryAsync(StableDiffusionGenerateRequest request, StableDiffusionModel activeModel, IProgress<double>? progress, CancellationToken ct)
        {
            Exception? lastException = null;

            for (int attempt = 0; attempt < 2; attempt++)
            {
                var inferenceBuffers = new List<Array>();

                try
                {
                    return await this.GenerateImageCoreAsync(request, activeModel, inferenceBuffers, progress, ct);
                }
                catch (Exception ex)
                {
                    lastException = ex;
                    this.ClearInferenceBuffers(inferenceBuffers);
                    await StaticLogger.LogAsync(ex, attempt == 0
                        ? "Error during inference. Cleared inference tensors from memory."
                        : "Error during inference after reload attempt.");

                    if (attempt == 0 && this.LoadedModel != null && this.LoadedModelOptions != null)
                    {
                        var reloadModel = this.LoadedModel;
                        var reloadOptions = this.LoadedModelOptions;

                        await StaticLogger.LogAsync("Reloading model as last resort after inference failure.");
                        await this.UnloadModelCoreAsync();
                        StableDiffusionModel? reloadedModel = await this.LoadModelCoreAsync(reloadModel, reloadOptions, this.CreateNestedProgress(progress, 0.0, 0.25), ct);
                        if (reloadedModel == null)
                        {
                            return null;
                        }
                    }
                }
                finally
                {
                    this.ClearInferenceBuffers(inferenceBuffers);
                }
            }

            if (lastException != null)
            {
                await StaticLogger.LogAsync(lastException, "Image generation failed.");
            }

            return null;
        }

        [SupportedOSPlatform("windows")]
        private async Task<ImageObj> GenerateImageCoreAsync(StableDiffusionGenerateRequest request, StableDiffusionModel activeModel, List<Array> inferenceBuffers, IProgress<double>? progress, CancellationToken ct)
        {
            if (this.TextEncoderSession == null || this.UnetSession == null || this.VaeDecoderSession == null || this.Tokenizer == null || this.Scheduler == null)
            {
                throw new InvalidOperationException("Stable Diffusion model resources are not fully loaded.");
            }

            ct.ThrowIfCancellationRequested();
            progress?.Report(0.25);
            int latentWidth = request.Width / 8;
            int latentHeight = request.Height / 8;

            int seed = request.Seed ?? this.CreateDeterministicSeed(request, activeModel);
            await StaticLogger.LogAsync($"Begin generation for model '{activeModel.ModelName}' size={request.Width}x{request.Height} steps={request.Steps} seed={seed}");

            TextEmbeddingsBuffer embeddings = await this.EncodePromptsAsync(request, inferenceBuffers, progress, ct);
            await StaticLogger.LogAsync($"Prompts encoded. SequenceLength={embeddings.SequenceLength} HiddenSize={embeddings.HiddenSize}");
            float[] latents = this.CreateInitialLatents(latentWidth, latentHeight, seed, inferenceBuffers);
            await StaticLogger.LogAsync($"Initial latents created (length={latents.Length}).");
            int[] timesteps = this.Scheduler.CreateTimesteps(request.Steps);

            progress?.Report(0.35);

            for (int stepIndex = 0; stepIndex < timesteps.Length; stepIndex++)
            {
                ct.ThrowIfCancellationRequested();

                int timestep = timesteps[stepIndex];
                int prevTimestep = stepIndex == timesteps.Length - 1 ? -1 : timesteps[stepIndex + 1];
                await StaticLogger.LogAsync($"Starting diffusion step {stepIndex + 1}/{timesteps.Length} (timestep={timestep})");

                float[] guidedNoise = await this.PredictNoiseAsync(latents, latentHeight, latentWidth, timestep, embeddings, (float) request.GuidanceScale, inferenceBuffers, ct);
                await StaticLogger.LogAsync($"Completed UNet prediction for step {stepIndex + 1}/{timesteps.Length}.");
                float[] nextLatents = this.Scheduler.Step(guidedNoise, latents, timestep, prevTimestep);
                inferenceBuffers.Add(nextLatents);

                Array.Clear(guidedNoise, 0, guidedNoise.Length);
                inferenceBuffers.Remove(guidedNoise);

                latents = this.ReplaceTrackedArray(inferenceBuffers, latents, nextLatents);
                progress?.Report(0.35 + (0.5 * ((stepIndex + 1) / (double) timesteps.Length)));
            }

            await StaticLogger.LogAsync("Decoding latents via VAE...");
            float[] decodedImage = await this.DecodeLatentsAsync(latents, request.Width, request.Height, inferenceBuffers, ct);
            await StaticLogger.LogAsync("VAE decoding complete.");
            progress?.Report(0.9);

            int outputWidth = request.Width;
            int outputHeight = request.Height;
            bool imageIsNormalizedZeroToOne = false;

            if (request.UseUpscaler && this.UpscalerSession != null)
            {
                await StaticLogger.LogAsync("Starting upscaling...");
                (float[] upscaledImage, int upscaledWidth, int upscaledHeight) = await this.UpscaleImageAsync(decodedImage, request.Width, request.Height, inferenceBuffers, ct);
                decodedImage = this.ReplaceTrackedArray(inferenceBuffers, decodedImage, upscaledImage);
                outputWidth = upscaledWidth;
                outputHeight = upscaledHeight;
                imageIsNormalizedZeroToOne = true;
                progress?.Report(0.95);
                await StaticLogger.LogAsync($"Upscaling complete. New size: {outputWidth}x{outputHeight}");
            }

            await StaticLogger.LogAsync("Encoding PNG...");
            byte[] pngData = await this.EncodePngAsync(decodedImage, outputWidth, outputHeight, imageIsNormalizedZeroToOne, ct);
            await StaticLogger.LogAsync("PNG encoding complete.");
            progress?.Report(1.0);

            return new ImageObj
            {
                FileName = $"{activeModel.ModelName}_{seed}_{outputWidth}x{outputHeight}.png",
                MediaType = "image/png",
                Width = outputWidth,
                Height = outputHeight,
                Data = pngData
            };
        }

        [SupportedOSPlatform("windows")]
        private async Task<TextEmbeddingsBuffer> EncodePromptsAsync(StableDiffusionGenerateRequest request, List<Array> inferenceBuffers, IProgress<double>? progress, CancellationToken ct)
        {
            TextEncoderEmbeddings primaryEmbeddings = await this.EncodePromptBatchAsync(this.TextEncoderSession!, this.Tokenizer!, request.Prompt, request.NegativePrompt, "TextEncoder", inferenceBuffers, ct);
            float[] embeddings = primaryEmbeddings.Data;
            int hiddenSize = primaryEmbeddings.HiddenSize;
            int sequenceLength = primaryEmbeddings.SequenceLength;
            bool shouldUseSecondaryEncoding = this.ShouldUseSecondaryTextEncoding();

            if (shouldUseSecondaryEncoding)
            {
                InferenceSession secondarySession = this.TextEncoderSession2 ?? this.TextEncoderSession!;
                ClipTokenizer secondaryTokenizer = this.Tokenizer2 ?? this.Tokenizer!;
                string secondaryEncoderName = this.TextEncoderSession2 != null || this.Tokenizer2 != null
                    ? "TextEncoder_2"
                    : "TextEncoder_2 (fallback primary)";

                TextEncoderEmbeddings secondaryEmbeddings = await this.EncodePromptBatchAsync(secondarySession, secondaryTokenizer, request.Prompt, request.NegativePrompt, secondaryEncoderName, inferenceBuffers, ct);
                if (secondaryEmbeddings.SequenceLength != sequenceLength)
                {
                    throw new InvalidOperationException("TextEncoder and TextEncoder_2 returned different sequence lengths.");
                }

                embeddings = this.ConcatEmbeddingsByHiddenDimension(primaryEmbeddings, secondaryEmbeddings, inferenceBuffers);
                hiddenSize += secondaryEmbeddings.HiddenSize;
                await StaticLogger.LogAsync($"Combined TextEncoder embeddings. PrimaryHidden={primaryEmbeddings.HiddenSize} SecondaryHidden={secondaryEmbeddings.HiddenSize} TotalHidden={hiddenSize}");
            }

            progress?.Report(0.3);

            return new TextEmbeddingsBuffer(embeddings, hiddenSize, sequenceLength);
        }

        private bool ShouldUseSecondaryTextEncoding()
        {
            StableDiffusionModel? model = this.LoadedModel;
            return this.TextEncoderSession2 != null
                || this.Tokenizer2 != null
                || !string.IsNullOrWhiteSpace(model?.TextEncoder2ModelOnnx)
                || !string.IsNullOrWhiteSpace(model?.Tokenizer2MergesTxt)
                || !string.IsNullOrWhiteSpace(model?.Tokenizer2VocabJson);
        }

        [SupportedOSPlatform("windows")]
        private async Task<TextEncoderEmbeddings> EncodePromptBatchAsync(InferenceSession session, ClipTokenizer tokenizer, string prompt, string negativePrompt, string encoderName, List<Array> inferenceBuffers, CancellationToken ct)
        {
            long[] negativeTokens = tokenizer.EncodeText(negativePrompt);
            long[] promptTokens = tokenizer.EncodeText(prompt);
            inferenceBuffers.Add(negativeTokens);
            inferenceBuffers.Add(promptTokens);

            long[] allTokens = [.. negativeTokens, .. promptTokens];
            inferenceBuffers.Add(allTokens);

            string inputName = this.FindInputName(session, "input_ids", "tokens");
            var inputs = this.CreateInputList(
                this.CreateIntegerTensorInput(inputName, session.InputMetadata[inputName].ElementType, allTokens, [2, tokenizer.MaxLength], inferenceBuffers));

            await StaticLogger.LogAsync($"Running {encoderName} session...");
            using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = await Task.Run(() => session.Run(inputs), ct);
            await StaticLogger.LogAsync($"{encoderName} run complete.");

            float[] embeddings = this.CopyTensorToFloatArray(results.First());
            inferenceBuffers.Add(embeddings);
            int hiddenSize = embeddings.Length / (2 * tokenizer.MaxLength);

            return new TextEncoderEmbeddings(embeddings, hiddenSize, tokenizer.MaxLength);
        }

        private float[] ConcatEmbeddingsByHiddenDimension(TextEncoderEmbeddings primaryEmbeddings, TextEncoderEmbeddings secondaryEmbeddings, List<Array> inferenceBuffers)
        {
            int batchSize = 2;
            int combinedHiddenSize = primaryEmbeddings.HiddenSize + secondaryEmbeddings.HiddenSize;
            var combinedEmbeddings = new float[batchSize * primaryEmbeddings.SequenceLength * combinedHiddenSize];
            inferenceBuffers.Add(combinedEmbeddings);

            for (int batchIndex = 0; batchIndex < batchSize; batchIndex++)
            {
                for (int tokenIndex = 0; tokenIndex < primaryEmbeddings.SequenceLength; tokenIndex++)
                {
                    int primarySourceIndex = ((batchIndex * primaryEmbeddings.SequenceLength) + tokenIndex) * primaryEmbeddings.HiddenSize;
                    int secondarySourceIndex = ((batchIndex * secondaryEmbeddings.SequenceLength) + tokenIndex) * secondaryEmbeddings.HiddenSize;
                    int destinationIndex = ((batchIndex * primaryEmbeddings.SequenceLength) + tokenIndex) * combinedHiddenSize;

                    Array.Copy(primaryEmbeddings.Data, primarySourceIndex, combinedEmbeddings, destinationIndex, primaryEmbeddings.HiddenSize);
                    Array.Copy(secondaryEmbeddings.Data, secondarySourceIndex, combinedEmbeddings, destinationIndex + primaryEmbeddings.HiddenSize, secondaryEmbeddings.HiddenSize);
                }
            }

            return combinedEmbeddings;
        }

        [SupportedOSPlatform("windows")]
        private float[] CreateInitialLatents(int latentWidth, int latentHeight, int seed, List<Array> inferenceBuffers)
        {
            int length = 4 * latentWidth * latentHeight;
            var latents = new float[length];
            var random = new Random(seed);

            for (int i = 0; i < length; i += 2)
            {
                double u1 = 1.0 - random.NextDouble();
                double u2 = 1.0 - random.NextDouble();
                double magnitude = Math.Sqrt(-2.0 * Math.Log(u1));
                double z0 = magnitude * Math.Cos(2.0 * Math.PI * u2);
                double z1 = magnitude * Math.Sin(2.0 * Math.PI * u2);

                latents[i] = (float) z0;
                if (i + 1 < length)
                {
                    latents[i + 1] = (float) z1;
                }
            }

            inferenceBuffers.Add(latents);
            return latents;
        }

        [SupportedOSPlatform("windows")]
        private async Task<float[]> PredictNoiseAsync(float[] latents, int latentHeight, int latentWidth, int timestep, TextEmbeddingsBuffer embeddings, float guidanceScale, List<Array> inferenceBuffers, CancellationToken ct)
        {
            int latentSampleLength = latents.Length;
            float[] latentModelInput = new float[latentSampleLength * 2];
            Array.Copy(latents, 0, latentModelInput, 0, latentSampleLength);
            Array.Copy(latents, 0, latentModelInput, latentSampleLength, latentSampleLength);
            inferenceBuffers.Add(latentModelInput);

            string sampleInputName = this.FindInputName(this.UnetSession!, "sample", "latent", "latent_model_input");
            string timestepInputName = this.FindInputName(this.UnetSession!, "timestep", "timesteps");
            string encoderInputName = this.FindInputName(this.UnetSession!, "encoder_hidden_states", "encoder_hidden_state", "context");

            var inputs = this.CreateInputList(
                this.CreateFloatTensorInput(sampleInputName, this.UnetSession!.InputMetadata[sampleInputName].ElementType, latentModelInput, [2, 4, latentHeight, latentWidth], inferenceBuffers),
                this.CreateTimestepInput(timestepInputName, this.UnetSession!.InputMetadata[timestepInputName].ElementType, timestep, inferenceBuffers),
                this.CreateFloatTensorInput(encoderInputName, this.UnetSession!.InputMetadata[encoderInputName].ElementType, embeddings.Data, [2, embeddings.SequenceLength, embeddings.HiddenSize], inferenceBuffers));

            await StaticLogger.LogAsync($"Running UNet session for timestep={timestep}, sampleShape=[2,4,{latentHeight},{latentWidth}]");
            using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = await Task.Run(() => this.UnetSession.Run(inputs), ct);
            await StaticLogger.LogAsync($"UNet session complete for timestep={timestep}");
            float[] noisePrediction = this.CopyTensorToFloatArray(results.First());
            inferenceBuffers.Add(noisePrediction);

            int singlePredictionLength = noisePrediction.Length / 2;
            var guidedNoise = new float[singlePredictionLength];
            inferenceBuffers.Add(guidedNoise);

            for (int i = 0; i < singlePredictionLength; i++)
            {
                float unconditional = noisePrediction[i];
                float conditional = noisePrediction[i + singlePredictionLength];
                guidedNoise[i] = unconditional + (guidanceScale * (conditional - unconditional));
            }

            Array.Clear(noisePrediction, 0, noisePrediction.Length);
            inferenceBuffers.Remove(noisePrediction);

            return guidedNoise;
        }

        [SupportedOSPlatform("windows")]
        private async Task<float[]> DecodeLatentsAsync(float[] latents, int width, int height, List<Array> inferenceBuffers, CancellationToken ct)
        {
            float[] scaledLatents = new float[latents.Length];
            for (int i = 0; i < latents.Length; i++)
            {
                scaledLatents[i] = latents[i] / 0.18215f;
            }

            inferenceBuffers.Add(scaledLatents);

            string inputName = this.FindInputName(this.VaeDecoderSession!, "latent_sample", "sample", "latent");
            var inputs = this.CreateInputList(
                this.CreateFloatTensorInput(inputName, this.VaeDecoderSession!.InputMetadata[inputName].ElementType, scaledLatents, [1, 4, height / 8, width / 8], inferenceBuffers));

            await StaticLogger.LogAsync($"Running VAE decoder for output {width}x{height}...");
            using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = await Task.Run(() => this.VaeDecoderSession.Run(inputs), ct);
            float[] decoded = this.CopyImageTensorToFloatArray(results.First(), width, height);
            await StaticLogger.LogAsync("VAE decoder run complete.");
            inferenceBuffers.Add(decoded);
            return decoded;
        }

        [SupportedOSPlatform("windows")]
        private async Task<(float[] Data, int Width, int Height)> UpscaleImageAsync(float[] decodedImage, int width, int height, List<Array> inferenceBuffers, CancellationToken ct)
        {
            if (this.UpscalerSession == null)
            {
                throw new InvalidOperationException("Upscaler model resources are not loaded.");
            }

            (int inputTileWidth, int inputTileHeight, int outputTileWidth, int outputTileHeight, int scaleX, int scaleY) = this.GetUpscalerDimensions();
            await StaticLogger.LogAsync($"Upscaler dimensions: inputTile={inputTileWidth}x{inputTileHeight} outputTile={outputTileWidth}x{outputTileHeight} scale={scaleX}x{scaleY}");
            int upscaledWidth = width * scaleX;
            int upscaledHeight = height * scaleY;
            var upscaledImage = new float[upscaledWidth * upscaledHeight * 3];
            inferenceBuffers.Add(upscaledImage);

            string inputName = this.FindInputName(this.UpscalerSession, "input", "image", "sample");

            for (int tileY = 0; tileY < height; tileY += inputTileHeight)
            {
                ct.ThrowIfCancellationRequested();

                int validInputHeight = Math.Min(inputTileHeight, height - tileY);
                for (int tileX = 0; tileX < width; tileX += inputTileWidth)
                {
                    int validInputWidth = Math.Min(inputTileWidth, width - tileX);
                    await StaticLogger.LogAsync($"Upscaler processing tile at ({tileX},{tileY}) size {validInputWidth}x{validInputHeight} -> output tile {outputTileWidth}x{outputTileHeight}");
                    float[] tileInput = this.CreateUpscalerTileInput(decodedImage, width, height, tileX, tileY, inputTileWidth, inputTileHeight);
                    inferenceBuffers.Add(tileInput);

                    var inputs = this.CreateInputList(
                        this.CreateFloatTensorInput(inputName, this.UpscalerSession.InputMetadata[inputName].ElementType, tileInput, [1, 3, inputTileHeight, inputTileWidth], inferenceBuffers));

                    using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = await Task.Run(() => this.UpscalerSession.Run(inputs), ct);
                    float[] tileOutput = this.CopyImageTensorToFloatArray(results.First(), outputTileWidth, outputTileHeight);
                    inferenceBuffers.Add(tileOutput);

                    this.CopyUpscaledTile(tileOutput, upscaledImage, upscaledWidth, upscaledHeight, tileX, tileY, validInputWidth, validInputHeight, outputTileWidth, outputTileHeight, scaleX, scaleY);

                    Array.Clear(tileOutput, 0, tileOutput.Length);
                    inferenceBuffers.Remove(tileOutput);

                    Array.Clear(tileInput, 0, tileInput.Length);
                    inferenceBuffers.Remove(tileInput);
                }
            }

            return (upscaledImage, upscaledWidth, upscaledHeight);
        }

        [SupportedOSPlatform("windows")]
        private async Task<byte[]> EncodePngAsync(float[] decodedImage, int width, int height, bool imageIsNormalizedZeroToOne, CancellationToken ct)
        {
            using var image = new Image<Rgba32>(width, height);
            int channelStride = width * height;

            for (int y = 0; y < height; y++)
            {
                ct.ThrowIfCancellationRequested();

                for (int x = 0; x < width; x++)
                {
                    int pixelIndex = (y * width) + x;
                    float r = imageIsNormalizedZeroToOne ? this.ClampZeroToOne(decodedImage[pixelIndex]) : this.ClampImageSample(decodedImage[pixelIndex]);
                    float g = imageIsNormalizedZeroToOne ? this.ClampZeroToOne(decodedImage[channelStride + pixelIndex]) : this.ClampImageSample(decodedImage[channelStride + pixelIndex]);
                    float b = imageIsNormalizedZeroToOne ? this.ClampZeroToOne(decodedImage[(2 * channelStride) + pixelIndex]) : this.ClampImageSample(decodedImage[(2 * channelStride) + pixelIndex]);
                    image[x, y] = new Rgba32((byte) (r * 255f), (byte) (g * 255f), (byte) (b * 255f), 255);
                }
            }

            using var memoryStream = new MemoryStream();
            await image.SaveAsPngAsync(memoryStream, new PngEncoder(), ct);
            return memoryStream.ToArray();
        }

        private float ClampImageSample(float value)
        {
            return Math.Clamp((value / 2f) + 0.5f, 0f, 1f);
        }

        private float ClampZeroToOne(float value)
        {
            return Math.Clamp(value, 0f, 1f);
        }

        private (int InputTileWidth, int InputTileHeight, int OutputTileWidth, int OutputTileHeight, int ScaleX, int ScaleY) GetUpscalerDimensions()
        {
            if (this.UpscalerSession == null)
            {
                throw new InvalidOperationException("Upscaler model resources are not loaded.");
            }

            int[] inputDimensions = this.UpscalerSession.InputMetadata.First().Value.Dimensions.ToArray();
            int[] outputDimensions = this.UpscalerSession.OutputMetadata.First().Value.Dimensions.ToArray();

            if (inputDimensions.Length != 4 || outputDimensions.Length != 4)
            {
                throw new InvalidOperationException("Unexpected upscaler tensor shape.");
            }

            if (inputDimensions[1] != 3 || outputDimensions[1] != 3)
            {
                throw new InvalidOperationException("Upscaler model must use 3 RGB channels.");
            }

            int inputTileHeight = inputDimensions[2];
            int inputTileWidth = inputDimensions[3];
            int outputTileHeight = outputDimensions[2];
            int outputTileWidth = outputDimensions[3];

            if (inputTileWidth <= 0 || inputTileHeight <= 0 || outputTileWidth <= 0 || outputTileHeight <= 0)
            {
                throw new InvalidOperationException("Upscaler model must expose fixed tile dimensions.");
            }

            if (outputTileWidth % inputTileWidth != 0 || outputTileHeight % inputTileHeight != 0)
            {
                throw new InvalidOperationException("Upscaler output dimensions must be multiples of the input dimensions.");
            }

            return (inputTileWidth, inputTileHeight, outputTileWidth, outputTileHeight, outputTileWidth / inputTileWidth, outputTileHeight / inputTileHeight);
        }

        private float[] CreateUpscalerTileInput(float[] decodedImage, int imageWidth, int imageHeight, int tileStartX, int tileStartY, int tileWidth, int tileHeight)
        {
            var tileInput = new float[3 * tileWidth * tileHeight];
            int sourceChannelStride = imageWidth * imageHeight;
            int tileChannelStride = tileWidth * tileHeight;

            for (int channel = 0; channel < 3; channel++)
            {
                int sourceChannelOffset = channel * sourceChannelStride;
                int tileChannelOffset = channel * tileChannelStride;

                for (int y = 0; y < tileHeight; y++)
                {
                    int sourceY = Math.Min(tileStartY + y, imageHeight - 1);
                    int sourceRowOffset = sourceChannelOffset + (sourceY * imageWidth);
                    int tileRowOffset = tileChannelOffset + (y * tileWidth);

                    for (int x = 0; x < tileWidth; x++)
                    {
                        int sourceX = Math.Min(tileStartX + x, imageWidth - 1);
                        tileInput[tileRowOffset + x] = this.ClampImageSample(decodedImage[sourceRowOffset + sourceX]);
                    }
                }
            }

            return tileInput;
        }

        private void CopyUpscaledTile(float[] tileOutput, float[] upscaledImage, int upscaledWidth, int upscaledHeight, int tileStartX, int tileStartY, int validInputWidth, int validInputHeight, int outputTileWidth, int outputTileHeight, int scaleX, int scaleY)
        {
            int upscaledChannelStride = upscaledWidth * upscaledHeight;
            int outputTileChannelStride = outputTileWidth * outputTileHeight;
            int validOutputWidth = validInputWidth * scaleX;
            int validOutputHeight = validInputHeight * scaleY;
            int destinationStartX = tileStartX * scaleX;
            int destinationStartY = tileStartY * scaleY;

            for (int channel = 0; channel < 3; channel++)
            {
                int destinationChannelOffset = channel * upscaledChannelStride;
                int sourceChannelOffset = channel * outputTileChannelStride;

                for (int y = 0; y < validOutputHeight; y++)
                {
                    int destinationRowOffset = destinationChannelOffset + ((destinationStartY + y) * upscaledWidth) + destinationStartX;
                    int sourceRowOffset = sourceChannelOffset + (y * outputTileWidth);
                    Array.Copy(tileOutput, sourceRowOffset, upscaledImage, destinationRowOffset, validOutputWidth);
                }
            }
        }

        private List<NamedOnnxValue> CreateInputList(params NamedOnnxValue[] values)
        {
            return [.. values];
        }

        private string FindInputName(InferenceSession session, params string[] preferredNames)
        {
            foreach (string preferred in preferredNames)
            {
                string? exactMatch = session.InputMetadata.Keys.FirstOrDefault(key => string.Equals(key, preferred, StringComparison.OrdinalIgnoreCase));
                if (exactMatch != null)
                {
                    return exactMatch;
                }

                string? containsMatch = session.InputMetadata.Keys.FirstOrDefault(key => key.Contains(preferred, StringComparison.OrdinalIgnoreCase));
                if (containsMatch != null)
                {
                    return containsMatch;
                }
            }

            return session.InputMetadata.Keys.First();
        }

        private NamedOnnxValue CreateIntegerTensorInput(string name, Type elementType, long[] data, int[] dimensions, List<Array> inferenceBuffers)
        {
            if (elementType == typeof(long))
            {
                return NamedOnnxValue.CreateFromTensor(name, new Microsoft.ML.OnnxRuntime.Tensors.DenseTensor<long>(data, dimensions));
            }

            if (elementType == typeof(int))
            {
                int[] converted = data.Select(static value => (int) value).ToArray();
                inferenceBuffers.Add(converted);
                return NamedOnnxValue.CreateFromTensor(name, new Microsoft.ML.OnnxRuntime.Tensors.DenseTensor<int>(converted, dimensions));
            }

            throw new NotSupportedException($"Unsupported integer tensor type '{elementType}'.");
        }

        private NamedOnnxValue CreateTimestepInput(string name, Type elementType, int timestep, List<Array> inferenceBuffers)
        {
            if (elementType == typeof(long))
            {
                long[] values = [timestep];
                inferenceBuffers.Add(values);
                return NamedOnnxValue.CreateFromTensor(name, new Microsoft.ML.OnnxRuntime.Tensors.DenseTensor<long>(values, [1]));
            }

            if (elementType == typeof(int))
            {
                int[] values = [timestep];
                inferenceBuffers.Add(values);
                return NamedOnnxValue.CreateFromTensor(name, new Microsoft.ML.OnnxRuntime.Tensors.DenseTensor<int>(values, [1]));
            }

            if (elementType == typeof(float))
            {
                float[] values = [timestep];
                inferenceBuffers.Add(values);
                return NamedOnnxValue.CreateFromTensor(name, new Microsoft.ML.OnnxRuntime.Tensors.DenseTensor<float>(values, [1]));
            }

            if (elementType == typeof(Half))
            {
                Half[] values = [(Half) timestep];
                inferenceBuffers.Add(values);
                return NamedOnnxValue.CreateFromTensor(name, new Microsoft.ML.OnnxRuntime.Tensors.DenseTensor<Half>(values, [1]));
            }

            if (elementType == typeof(double))
            {
                double[] values = [timestep];
                inferenceBuffers.Add(values);
                return NamedOnnxValue.CreateFromTensor(name, new Microsoft.ML.OnnxRuntime.Tensors.DenseTensor<double>(values, [1]));
            }

            throw new NotSupportedException($"Unsupported timestep tensor type '{elementType}'.");
        }

        private NamedOnnxValue CreateFloatTensorInput(string name, Type elementType, float[] data, int[] dimensions, List<Array> inferenceBuffers)
        {
            if (elementType == typeof(float))
            {
                return NamedOnnxValue.CreateFromTensor(name, new Microsoft.ML.OnnxRuntime.Tensors.DenseTensor<float>(data, dimensions));
            }

            if (elementType == typeof(Half))
            {
                Half[] converted = data.Select(static value => (Half) value).ToArray();
                inferenceBuffers.Add(converted);
                return NamedOnnxValue.CreateFromTensor(name, new Microsoft.ML.OnnxRuntime.Tensors.DenseTensor<Half>(converted, dimensions));
            }

            if (elementType == typeof(double))
            {
                double[] converted = data.Select(static value => (double) value).ToArray();
                inferenceBuffers.Add(converted);
                return NamedOnnxValue.CreateFromTensor(name, new Microsoft.ML.OnnxRuntime.Tensors.DenseTensor<double>(converted, dimensions));
            }

            throw new NotSupportedException($"Unsupported floating-point tensor type '{elementType}'.");
        }

        private float[] CopyTensorToFloatArray(DisposableNamedOnnxValue value)
        {
            if (value.Value is Microsoft.ML.OnnxRuntime.Tensors.Tensor<float> floatTensor)
            {
                return [.. floatTensor];
            }

            if (value.Value is Microsoft.ML.OnnxRuntime.Tensors.Tensor<Half> halfTensor)
            {
                return [.. halfTensor.Select(static item => (float) item)];
            }

            if (value.Value is Microsoft.ML.OnnxRuntime.Tensors.Tensor<double> doubleTensor)
            {
                return [.. doubleTensor.Select(static item => (float) item)];
            }

            throw new NotSupportedException($"Unsupported tensor output type '{value.Value?.GetType().FullName}'.");
        }

        private float[] CopyImageTensorToFloatArray(DisposableNamedOnnxValue value, int width, int height)
        {
            if (value.Value is Microsoft.ML.OnnxRuntime.Tensors.Tensor<float> floatTensor)
            {
                return this.NormalizeImageTensor([.. floatTensor], width, height);
            }

            if (value.Value is Microsoft.ML.OnnxRuntime.Tensors.Tensor<Half> halfTensor)
            {
                return this.NormalizeImageTensor([.. halfTensor.Select(static item => (float) item)], width, height);
            }

            if (value.Value is Microsoft.ML.OnnxRuntime.Tensors.Tensor<double> doubleTensor)
            {
                return this.NormalizeImageTensor([.. doubleTensor.Select(static item => (float) item)], width, height);
            }

            throw new NotSupportedException($"Unsupported image tensor output type '{value.Value?.GetType().FullName}'.");
        }

        private float[] NormalizeImageTensor(float[] data, int width, int height)
        {
            if (data.Length == width * height * 3)
            {
                return data;
            }

            throw new InvalidOperationException("Unexpected VAE decoder output shape.");
        }

        private float[] ReplaceTrackedArray(List<Array> inferenceBuffers, float[] oldValue, float[] newValue)
        {
            Array.Clear(oldValue, 0, oldValue.Length);
            inferenceBuffers.Remove(oldValue);
            return newValue;
        }

        private int CreateDeterministicSeed(StableDiffusionGenerateRequest request, StableDiffusionModel activeModel)
        {
            string seedSource = $"{activeModel.ModelName}|{request.Prompt}|{request.NegativePrompt}|{request.Width}|{request.Height}|{request.Steps}|{request.GuidanceScale}|{request.Strength}";
            return Math.Abs(seedSource.GetHashCode());
        }

        private void ClearInferenceBuffers(List<Array> inferenceBuffers)
        {
            foreach (Array buffer in inferenceBuffers)
            {
                Array.Clear(buffer, 0, buffer.Length);
            }

            inferenceBuffers.Clear();
        }

        private IProgress<double>? CreateNestedProgress(IProgress<double>? progress, double start, double end)
        {
            if (progress == null)
            {
                return null;
            }

            return new Progress<double>(value =>
            {
                double clamped = Math.Clamp(value, 0.0, 1.0);
                progress.Report(start + ((end - start) * clamped));
            });
        }

        private sealed record TextEmbeddingsBuffer(float[] Data, int HiddenSize, int SequenceLength);
        private sealed record TextEncoderEmbeddings(float[] Data, int HiddenSize, int SequenceLength);
    }
}
