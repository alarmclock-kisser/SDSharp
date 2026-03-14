using Microsoft.ML.OnnxRuntime;
using SDSharp.Shared;
using System.Text.Json;

namespace SDSharp.Onnx
{
    public partial class OnnxService
    {
        public StableDiffusionModel? LoadedModel { get; private set; } = null;
        public StableDiffusionLoadOptions? LoadedModelOptions { get; private set; } = null;

        private InferenceSession? TextEncoderSession;
        private InferenceSession? UnetSession;
        private InferenceSession? VaeDecoderSession;
        private InferenceSession? VaeEncoderSession;
        private InferenceSession? UpscalerSession;
        private JsonDocument? SchedulerConfigDocument;
        private JsonDocument? UpscalerConfigDocument;
        private StableDiffusionScheduler? Scheduler;
        private ClipTokenizer? Tokenizer;
        private string? TokenizerMergesContent;
        private string? TokenizerVocabContent;



        public async Task<StableDiffusionModel?> LoadModelAsync(StableDiffusionModel model, StableDiffusionLoadOptions? options = null, IProgress<double>? progress = null, CancellationToken ct = default)
        {
            options ??= new StableDiffusionLoadOptions();

            await this.StateLock.WaitAsync(ct);
            try
            {
                return await this.LoadModelCoreAsync(model, options, progress, ct);
            }
            finally
            {
                this.StateLock.Release();
            }
        }



        public async Task<bool?> UnloadModelAsync()
        {
            await this.StateLock.WaitAsync();
            try
            {
                return await this.UnloadModelCoreAsync();
            }
            finally
            {
                this.StateLock.Release();
            }
        }



        private async Task<StableDiffusionModel?> LoadModelCoreAsync(StableDiffusionModel model, StableDiffusionLoadOptions options, IProgress<double>? progress, CancellationToken ct)
        {
            ct.ThrowIfCancellationRequested();
            progress?.Report(0.0);

            model.PopulateMissingPathsFromRootDirectory();

            var validationErrors = model.Validate();
            validationErrors.AddRange(options.Validate());

            if (validationErrors.Count > 0)
            {
                foreach (var error in validationErrors.Distinct())
                {
                    StaticLogger.Log($"Load validation failed: {error}");
                }

                return null;
            }

            if (this.LoadedModel?.IsSameModelAs(model) == true)
            {
                progress?.Report(1.0);
                await StaticLogger.LogAsync($"Model '{this.LoadedModel.ModelName}' is already loaded.");
                return this.LoadedModel;
            }

            if (this.LoadedModel != null)
            {
                if (!options.ForceUnload)
                {
                    progress?.Report(1.0);
                    await StaticLogger.LogAsync($"Model '{this.LoadedModel.ModelName}' remains loaded because ForceUnload is false.");
                    return this.LoadedModel;
                }

                var unloadResult = await this.UnloadModelCoreAsync();
                if (unloadResult == false)
                {
                    progress?.Report(1.0);
                    return this.LoadedModel;
                }
            }

            var resolvedOptions = this.ResolveLoadOptions(options.Normalize());

            try
            {
                await this.LoadModelResourcesAsync(model, resolvedOptions, progress, ct);
            }
            catch (Exception ex) when (string.Equals(resolvedOptions.OrtExecutionProvider, "Dml", StringComparison.OrdinalIgnoreCase))
            {
                StaticLogger.Log(ex, $"Error loading model '{model.ModelName}' with DML. Falling back to CPU.");
                this.DisposeModelResources();

                resolvedOptions.OrtExecutionProvider = "Cpu";
                resolvedOptions.DirectMlDeviceId = -1;

                await this.LoadModelResourcesAsync(model, resolvedOptions, progress, ct);
            }

            this.LoadedModel = model;
            this.LoadedModelOptions = resolvedOptions;
            this.AvailableModels = this.GetAvailableModels();

            progress?.Report(1.0);
            await StaticLogger.LogAsync($"Model '{model.ModelName}' loaded successfully using {resolvedOptions.OrtExecutionProvider}.");

            return this.LoadedModel;
        }



        private async Task LoadModelResourcesAsync(StableDiffusionModel model, StableDiffusionLoadOptions options, IProgress<double>? progress, CancellationToken ct)
        {
            ct.ThrowIfCancellationRequested();
            progress?.Report(0.1);

            this.DisposeModelResources();

            await StaticLogger.LogAsync($"Loading scheduler config from '{model.SchedulerConfigJson}'...");
            this.SchedulerConfigDocument = JsonDocument.Parse(await File.ReadAllTextAsync(model.SchedulerConfigJson!, ct));
            await StaticLogger.LogAsync("Scheduler config loaded.");
            this.TokenizerMergesContent = await File.ReadAllTextAsync(model.TokenizerMergesTxt!, ct);
            await StaticLogger.LogAsync($"Tokenizer merges loaded from '{model.TokenizerMergesTxt}'.");
            this.TokenizerVocabContent = await File.ReadAllTextAsync(model.TokenizerVocabJson!, ct);
            await StaticLogger.LogAsync($"Tokenizer vocab loaded from '{model.TokenizerVocabJson}'.");
            this.Scheduler = StableDiffusionScheduler.Create(this.SchedulerConfigDocument.RootElement);
            await StaticLogger.LogAsync("Scheduler created.");
            this.Tokenizer = new ClipTokenizer(this.TokenizerVocabContent, this.TokenizerMergesContent);
            await StaticLogger.LogAsync("Tokenizer created.");

            progress?.Report(0.2);

            await StaticLogger.LogAsync($"Creating TextEncoder session from '{model.TextEncoderModelOnnx}'...");
            this.TextEncoderSession = await this.CreateInferenceSessionAsync(model.TextEncoderModelOnnx!, options, ct);
            await StaticLogger.LogAsync("TextEncoder session created.");
            progress?.Report(0.45);

            await StaticLogger.LogAsync($"Creating Unet session from '{model.UnetModelOnnx}'...");
            this.UnetSession = await this.CreateInferenceSessionAsync(model.UnetModelOnnx!, options, ct);
            await StaticLogger.LogAsync("Unet session created.");
            progress?.Report(0.7);

            await StaticLogger.LogAsync($"Creating VAE decoder session from '{model.VaeDecoderModelOnnx}'...");
            this.VaeDecoderSession = await this.CreateInferenceSessionAsync(model.VaeDecoderModelOnnx!, options, ct);
            await StaticLogger.LogAsync("VAE decoder session created.");
            progress?.Report(0.9);

            if (!string.IsNullOrWhiteSpace(model.VaeEncoderModelOnnx))
            {
                await StaticLogger.LogAsync($"Creating VAE encoder session from '{model.VaeEncoderModelOnnx}'...");
                this.VaeEncoderSession = await this.CreateInferenceSessionAsync(model.VaeEncoderModelOnnx!, options, ct);
                await StaticLogger.LogAsync("VAE encoder session created.");
            }

            if (!string.IsNullOrWhiteSpace(model.UpscalerConfigJson) && !string.IsNullOrWhiteSpace(model.UpscalerModelOnnx))
            {
                await StaticLogger.LogAsync($"Loading upscaler config from '{model.UpscalerConfigJson}'...");
                this.UpscalerConfigDocument = JsonDocument.Parse(await File.ReadAllTextAsync(model.UpscalerConfigJson!, ct));
                await StaticLogger.LogAsync("Upscaler config loaded.");

                await StaticLogger.LogAsync($"Creating Upscaler session from '{model.UpscalerModelOnnx}'...");
                this.UpscalerSession = await this.CreateInferenceSessionAsync(model.UpscalerModelOnnx!, options, ct);
                await StaticLogger.LogAsync("Upscaler session created.");
            }
        }



        private async Task<InferenceSession> CreateInferenceSessionAsync(string modelPath, StableDiffusionLoadOptions options, CancellationToken ct)
        {
            return await Task.Run(() =>
            {
                using var sessionOptions = this.CreateSessionOptions(options);
                return new InferenceSession(modelPath, sessionOptions);
            }, ct);
        }



        private SessionOptions CreateSessionOptions(StableDiffusionLoadOptions options)
        {
            var sessionOptions = new SessionOptions
            {
                ExecutionMode = this.ParseExecutionMode(options.OrtExecutionMode),
                GraphOptimizationLevel = this.ParseGraphOptimizationLevel(options.OrtGraphOptimizationLevel)
            };

            if (string.Equals(options.OrtExecutionProvider, "Dml", StringComparison.OrdinalIgnoreCase) && options.DirectMlDeviceId >= 0)
            {
                sessionOptions.AppendExecutionProvider_DML(options.DirectMlDeviceId);
            }

            return sessionOptions;
        }



        private StableDiffusionLoadOptions ResolveLoadOptions(StableDiffusionLoadOptions options)
        {
            if (string.Equals(options.OrtExecutionProvider, "Dml", StringComparison.OrdinalIgnoreCase))
            {
                if (this.DirectMlDeviceNames.Count == 0 || (this.DirectMlDeviceNames.Count == 1 && string.Equals(this.DirectMlDeviceNames[0], "CPU", StringComparison.OrdinalIgnoreCase)))
                {
                    options.OrtExecutionProvider = "Cpu";
                    options.DirectMlDeviceId = -1;
                    return options;
                }

                if (options.DirectMlDeviceId < 0)
                {
                    options.DirectMlDeviceId = 0;
                }
            }

            return options;
        }



        private ExecutionMode ParseExecutionMode(string executionMode)
        {
            return string.Equals(executionMode, "Parallel", StringComparison.OrdinalIgnoreCase)
                ? ExecutionMode.ORT_PARALLEL
                : ExecutionMode.ORT_SEQUENTIAL;
        }



        private GraphOptimizationLevel ParseGraphOptimizationLevel(string graphOptimizationLevel)
        {
            if (string.Equals(graphOptimizationLevel, "Basic", StringComparison.OrdinalIgnoreCase))
            {
                return GraphOptimizationLevel.ORT_ENABLE_BASIC;
            }

            if (string.Equals(graphOptimizationLevel, "Extended", StringComparison.OrdinalIgnoreCase))
            {
                return GraphOptimizationLevel.ORT_ENABLE_EXTENDED;
            }

            if (string.Equals(graphOptimizationLevel, "All", StringComparison.OrdinalIgnoreCase))
            {
                return GraphOptimizationLevel.ORT_ENABLE_ALL;
            }

            return GraphOptimizationLevel.ORT_DISABLE_ALL;
        }



        private async Task<bool?> UnloadModelCoreAsync()
        {
            if (this.LoadedModel == null)
            {
                await StaticLogger.LogAsync("No model is currently loaded.");
                return null;
            }

            string modelName = this.LoadedModel.ModelName;

            try
            {
                this.DisposeModelResources();
                this.LoadedModel = null;
                this.LoadedModelOptions = null;

                GC.Collect();
                GC.WaitForPendingFinalizers();

                await StaticLogger.LogAsync($"Model '{modelName}' unloaded successfully.");
                return true;
            }
            catch (Exception ex)
            {
                StaticLogger.Log(ex, $"Error unloading model '{modelName}'.");
                return this.LoadedModel == null ? true : false;
            }
        }



        private void DisposeModelResources()
        {
            this.TextEncoderSession?.Dispose();
            this.TextEncoderSession = null;

            this.UnetSession?.Dispose();
            this.UnetSession = null;

            this.VaeDecoderSession?.Dispose();
            this.VaeDecoderSession = null;

            this.VaeEncoderSession?.Dispose();
            this.VaeEncoderSession = null;

            this.UpscalerSession?.Dispose();
            this.UpscalerSession = null;

            this.SchedulerConfigDocument?.Dispose();
            this.SchedulerConfigDocument = null;

            this.UpscalerConfigDocument?.Dispose();
            this.UpscalerConfigDocument = null;

            this.Scheduler = null;
            this.Tokenizer = null;

            this.TokenizerMergesContent = null;
            this.TokenizerVocabContent = null;
        }



    }
}
