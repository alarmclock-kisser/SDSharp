using System.Text.Json;

namespace SDSharp.Shared
{
    public class StableDiffusionModel
    {
        public string ModelRootDirectory { get; set; } = string.Empty;
        public string ModelName { get; set; } = string.Empty;

        public string? SchedulerConfigJson { get; set; }
        public string? TextEncoderModelOnnx { get; set; }
        public string? TextEncoder2ModelOnnx { get; set; }
        public string? TokenizerMergesTxt { get; set; }
        public string? TokenizerVocabJson { get; set; }
        public string? Tokenizer2MergesTxt { get; set; }
        public string? Tokenizer2VocabJson { get; set; }
        public string? UnetModelOnnx { get; set; }
        public string? UnetWeightsPb { get; set; }
        public string? VaeDecoderModelOnnx { get; set; }
        public string? VaeEncoderModelOnnx { get; set; }
        public string? UpscalerConfigJson { get; set; }
        public string? UpscalerModelOnnx { get; set; }

        public double ModelSizeInMb { get; set; } = 0.0;



        public StableDiffusionModel()
        {
        }



        public StableDiffusionModel(string modelRootDirectory)
        {
            this.LoadFromRootDirectory(modelRootDirectory);
        }



        public void LoadFromRootDirectory(string modelRootDirectory)
        {
            if (!Directory.Exists(modelRootDirectory))
            {
                throw new DirectoryNotFoundException($"Model root directory '{modelRootDirectory}' does not exist.");
            }

            this.ModelRootDirectory = modelRootDirectory;
            this.ModelName = Path.GetFileName(modelRootDirectory.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar));

            try
            {
                this.PopulateMissingPathsFromRootDirectory();
                this.ModelSizeInMb = Directory.GetFiles(modelRootDirectory, "*.*", SearchOption.AllDirectories)
                    .Select(f => new FileInfo(f).Length)
                    .Sum() / (1024.0 * 1024.0);
            }
            catch (Exception ex)
            {
                StaticLogger.Log($"Error loading model files for '{modelRootDirectory}': {ex.Message}");
            }
        }



        public void PopulateMissingPathsFromRootDirectory()
        {
            if (string.IsNullOrWhiteSpace(this.ModelRootDirectory) || !Directory.Exists(this.ModelRootDirectory))
            {
                return;
            }

            this.ModelName = string.IsNullOrWhiteSpace(this.ModelName)
                ? Path.GetFileName(this.ModelRootDirectory.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar))
                : this.ModelName;

            this.SchedulerConfigJson ??= this.TryGetFirstFile("scheduler", "*scheduler_config.json");
            this.TextEncoderModelOnnx ??= this.TryGetFirstFile("text_encoder", "*.onnx");
            this.TextEncoder2ModelOnnx ??= this.TryGetFirstFile("text_encoder_2", "*.onnx");
            this.TokenizerMergesTxt ??= this.TryGetFirstFile("tokenizer", "*merges.txt");
            this.TokenizerVocabJson ??= this.TryGetFirstFile("tokenizer", "*vocab.json");
            this.Tokenizer2MergesTxt ??= this.TryGetFirstFile("tokenizer_2", "*merges.txt");
            this.Tokenizer2VocabJson ??= this.TryGetFirstFile("tokenizer_2", "*vocab.json");
            this.UnetModelOnnx ??= this.TryGetFirstFile("unet", "*.onnx");
            this.UnetWeightsPb ??= this.TryGetFirstFile("unet", "*.pb");
            this.VaeDecoderModelOnnx ??= this.TryGetFirstFile("vae_decoder", "*.onnx");
            this.VaeEncoderModelOnnx ??= this.TryGetFirstFile("vae_encoder", "*.onnx");

            string? upscalerDir = this.FindUpscalerDirectory();
            if (!string.IsNullOrEmpty(upscalerDir))
            {
                this.UpscalerConfigJson ??= this.TryGetFirstFile(upscalerDir, "*config.json");
                this.UpscalerModelOnnx ??= this.TryGetFirstFile(upscalerDir, "*.onnx");
            }

            this.ModelSizeInMb = Directory.GetFiles(this.ModelRootDirectory, "*.*", SearchOption.AllDirectories)
                .Select(f => new FileInfo(f).Length)
                .Sum() / (1024.0 * 1024.0);
        }



        public List<string> Validate()
        {
            this.PopulateMissingPathsFromRootDirectory();

            var errors = new List<string>();
            string? upscalerDir = this.FindUpscalerDirectory();
            bool hasUpscaler = !string.IsNullOrWhiteSpace(upscalerDir)
                || !string.IsNullOrWhiteSpace(this.UpscalerConfigJson)
                || !string.IsNullOrWhiteSpace(this.UpscalerModelOnnx);
            if (string.IsNullOrWhiteSpace(this.ModelRootDirectory))
            {
                errors.Add("ModelRootDirectory is required.");
            }
            else if (!Directory.Exists(this.ModelRootDirectory))
            {
                errors.Add($"ModelRootDirectory '{this.ModelRootDirectory}' does not exist.");
            }

            if (string.IsNullOrWhiteSpace(this.ModelName))
            {
                errors.Add("ModelName is required.");
            }

            ValidateRequiredFile(errors, this.SchedulerConfigJson, nameof(this.SchedulerConfigJson));
            ValidateRequiredFile(errors, this.TextEncoderModelOnnx, nameof(this.TextEncoderModelOnnx));
            ValidateRequiredFile(errors, this.TokenizerMergesTxt, nameof(this.TokenizerMergesTxt));
            ValidateRequiredFile(errors, this.TokenizerVocabJson, nameof(this.TokenizerVocabJson));
            ValidateOptionalFile(errors, this.TextEncoder2ModelOnnx, nameof(this.TextEncoder2ModelOnnx));
            ValidateOptionalFile(errors, this.Tokenizer2MergesTxt, nameof(this.Tokenizer2MergesTxt));
            ValidateOptionalFile(errors, this.Tokenizer2VocabJson, nameof(this.Tokenizer2VocabJson));

            ValidateRequiredFile(errors, this.UnetModelOnnx, nameof(this.UnetModelOnnx));
            ValidateRequiredFile(errors, this.VaeDecoderModelOnnx, nameof(this.VaeDecoderModelOnnx));
            ValidateOptionalFile(errors, this.UnetWeightsPb, nameof(this.UnetWeightsPb));
            ValidateOptionalFile(errors, this.VaeEncoderModelOnnx, nameof(this.VaeEncoderModelOnnx));

            if (hasUpscaler)
            {
                ValidateRequiredFile(errors, this.UpscalerConfigJson, nameof(this.UpscalerConfigJson));
                ValidateRequiredFile(errors, this.UpscalerModelOnnx, nameof(this.UpscalerModelOnnx));
            }

            if (!string.IsNullOrWhiteSpace(this.SchedulerConfigJson) && File.Exists(this.SchedulerConfigJson))
            {
                try
                {
                    using var schedulerDocument = JsonDocument.Parse(File.ReadAllText(this.SchedulerConfigJson));
                    _ = schedulerDocument.RootElement.ValueKind;
                }
                catch (Exception ex)
                {
                    errors.Add($"SchedulerConfigJson '{this.SchedulerConfigJson}' is not valid JSON: {ex.Message}");
                }
            }

            if (!string.IsNullOrWhiteSpace(this.Tokenizer2VocabJson) && File.Exists(this.Tokenizer2VocabJson))
            {
                try
                {
                    using var vocab2Document = JsonDocument.Parse(File.ReadAllText(this.Tokenizer2VocabJson));
                    _ = vocab2Document.RootElement.ValueKind;
                }
                catch (Exception ex)
                {
                    errors.Add($"Tokenizer2VocabJson '{this.Tokenizer2VocabJson}' is not valid JSON: {ex.Message}");
                }
            }

            if (!string.IsNullOrWhiteSpace(this.TokenizerVocabJson) && File.Exists(this.TokenizerVocabJson))
            {
                try
                {
                    using var vocabDocument = JsonDocument.Parse(File.ReadAllText(this.TokenizerVocabJson));
                    _ = vocabDocument.RootElement.ValueKind;
                }
                catch (Exception ex)
                {
                    errors.Add($"TokenizerVocabJson '{this.TokenizerVocabJson}' is not valid JSON: {ex.Message}");
                }
            }

            if (!string.IsNullOrWhiteSpace(this.UpscalerConfigJson) && File.Exists(this.UpscalerConfigJson))
            {
                try
                {
                    using var upscalerDocument = JsonDocument.Parse(File.ReadAllText(this.UpscalerConfigJson));
                    _ = upscalerDocument.RootElement.ValueKind;
                }
                catch (Exception ex)
                {
                    errors.Add($"UpscalerConfigJson '{this.UpscalerConfigJson}' is not valid JSON: {ex.Message}");
                }
            }

            return errors;
        }



        public bool IsSameModelAs(StableDiffusionModel? other)
        {
            if (other == null)
            {
                return false;
            }

            return string.Equals(Path.GetFullPath(this.ModelRootDirectory), Path.GetFullPath(other.ModelRootDirectory), StringComparison.OrdinalIgnoreCase)
                && string.Equals(this.ModelName, other.ModelName, StringComparison.OrdinalIgnoreCase);
        }



        private string? TryGetFirstFile(string subDirectory, string pattern)
        {
            string fullDirectory = Path.Combine(this.ModelRootDirectory, subDirectory);
            return Directory.Exists(fullDirectory)
                ? Directory.GetFiles(fullDirectory, pattern, SearchOption.TopDirectoryOnly).FirstOrDefault()
                : null;
        }



        private string? FindUpscalerDirectory()
        {
            if (string.IsNullOrWhiteSpace(this.ModelRootDirectory) || !Directory.Exists(this.ModelRootDirectory))
            {
                return null;
            }

            return Directory.GetDirectories(this.ModelRootDirectory, "*upscaler*", SearchOption.AllDirectories)
                .OrderBy(static path => path.Length)
                .FirstOrDefault();
        }



        private static void ValidateRequiredFile(List<string> errors, string? filePath, string propertyName)
        {
            if (string.IsNullOrWhiteSpace(filePath))
            {
                errors.Add($"{propertyName} is required.");
                return;
            }

            if (!File.Exists(filePath))
            {
                errors.Add($"{propertyName} file '{filePath}' does not exist.");
            }
        }



        private static void ValidateOptionalFile(List<string> errors, string? filePath, string propertyName)
        {
            if (!string.IsNullOrWhiteSpace(filePath) && !File.Exists(filePath))
            {
                errors.Add($"{propertyName} file '{filePath}' does not exist.");
            }
        }
    }
}
