using System.ComponentModel.DataAnnotations;

namespace SDSharp.Shared
{
    public class StableDiffusionLoadRequest
    {
        public StableDiffusionModel Model { get; set; } = new();
        public StableDiffusionLoadOptions Options { get; set; } = new();
        public bool LogProgress { get; set; } = true;



        public List<string> Validate()
        {
            var errors = new List<string>();

            if (this.Model == null)
            {
                errors.Add("Model is required.");
            }
            else
            {
                errors.AddRange(this.Model.Validate());
            }

            if (this.Options != null)
            {
                errors.AddRange(this.Options.Validate());
            }

            return errors;
        }
    }



    public class StableDiffusionGenerateRequest
    {
        public StableDiffusionModel? Model { get; set; }
        public StableDiffusionLoadOptions? LoadOptions { get; set; }
        public bool AutoLoadModel { get; set; } = true;
        public bool LogProgress { get; set; } = true;
        public bool UseUpscaler { get; set; } = true;

        public string Prompt { get; set; } = "a beautiful landscape, scenic mountains, natural light, highly detailed";
        public string NegativePrompt { get; set; } = "blurry, low quality, distorted, artifacts";

        [Range(64, 2048)]
        public int Width { get; set; } = 512;

        [Range(64, 2048)]
        public int Height { get; set; } = 512;

        [Range(1, 150)]
        public int Steps { get; set; } = 16;

        public int? Seed { get; set; }

        [Range(1.0, 30.0)]
        public double GuidanceScale { get; set; } = 7.5;

        [Range(0.0, 1.0)]
        public double Strength { get; set; } = 1.0;



        public List<string> Validate()
        {
            var errors = new List<string>();

            if (string.IsNullOrWhiteSpace(this.Prompt))
            {
                errors.Add("Prompt is required.");
            }

            if (this.Width % 8 != 0)
            {
                errors.Add("Width must be divisible by 8.");
            }

            if (this.Height % 8 != 0)
            {
                errors.Add("Height must be divisible by 8.");
            }

            if (this.Width < 64 || this.Width > 2048)
            {
                errors.Add("Width must be between 64 and 2048.");
            }

            if (this.Height < 64 || this.Height > 2048)
            {
                errors.Add("Height must be between 64 and 2048.");
            }

            if (this.Steps < 1 || this.Steps > 150)
            {
                errors.Add("Steps must be between 1 and 150.");
            }

            if (this.GuidanceScale < 1.0 || this.GuidanceScale > 30.0)
            {
                errors.Add("GuidanceScale must be between 1.0 and 30.0.");
            }

            if (this.Strength < 0.0 || this.Strength > 1.0)
            {
                errors.Add("Strength must be between 0.0 and 1.0.");
            }

            if (this.Model != null)
            {
                errors.AddRange(this.Model.Validate());
            }

            if (this.LoadOptions != null)
            {
                errors.AddRange(this.LoadOptions.Validate());
            }

            return errors;
        }
    }



    public class ImageObj
    {
        public string FileName { get; set; } = "image.png";
        public string MediaType { get; set; } = "image/png";
        public int Width { get; set; }
        public int Height { get; set; }
        public byte[] Data { get; set; } = [];

        public string Base64 => Convert.ToBase64String(this.Data);
        public string DataUrl => $"data:{this.MediaType};base64,{this.Base64}";
    }
}
