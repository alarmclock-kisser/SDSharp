using System;
using System.Collections.Generic;
using System.Text;

namespace SDSharp.Shared
{
    public class StableDiffusionLoadOptions
    {
        public bool ForceUnload { get; set; } = true;

        public int DirectMlDeviceId { get; set; } = -1; // -1 for CPU

        public string OrtExecutionProvider { get; set; } = "Dml"; // "Dml" or "Cpu"
        public string OrtGraphOptimizationLevel { get; set; } = "DisableAll"; // "DisableAll", "Basic", "Extended", "All"
        public string OrtExecutionMode { get; set; } = "Sequential"; // "Sequential" or "Parallel"



        public StableDiffusionLoadOptions Normalize()
        {
            return new StableDiffusionLoadOptions
            {
                ForceUnload = this.ForceUnload,
                DirectMlDeviceId = this.DirectMlDeviceId,
                OrtExecutionProvider = string.IsNullOrWhiteSpace(this.OrtExecutionProvider) ? "Dml" : this.OrtExecutionProvider.Trim(),
                OrtGraphOptimizationLevel = string.IsNullOrWhiteSpace(this.OrtGraphOptimizationLevel) ? "DisableAll" : this.OrtGraphOptimizationLevel.Trim(),
                OrtExecutionMode = string.IsNullOrWhiteSpace(this.OrtExecutionMode) ? "Sequential" : this.OrtExecutionMode.Trim()
            };
        }



        public List<string> Validate()
        {
            var normalized = this.Normalize();
            var errors = new List<string>();

            if (!string.Equals(normalized.OrtExecutionProvider, "Dml", StringComparison.OrdinalIgnoreCase)
                && !string.Equals(normalized.OrtExecutionProvider, "Cpu", StringComparison.OrdinalIgnoreCase))
            {
                errors.Add("OrtExecutionProvider must be either 'Dml' or 'Cpu'.");
            }

            if (!string.Equals(normalized.OrtGraphOptimizationLevel, "DisableAll", StringComparison.OrdinalIgnoreCase)
                && !string.Equals(normalized.OrtGraphOptimizationLevel, "Basic", StringComparison.OrdinalIgnoreCase)
                && !string.Equals(normalized.OrtGraphOptimizationLevel, "Extended", StringComparison.OrdinalIgnoreCase)
                && !string.Equals(normalized.OrtGraphOptimizationLevel, "All", StringComparison.OrdinalIgnoreCase))
            {
                errors.Add("OrtGraphOptimizationLevel must be one of: DisableAll, Basic, Extended, All.");
            }

            if (!string.Equals(normalized.OrtExecutionMode, "Sequential", StringComparison.OrdinalIgnoreCase)
                && !string.Equals(normalized.OrtExecutionMode, "Parallel", StringComparison.OrdinalIgnoreCase))
            {
                errors.Add("OrtExecutionMode must be either 'Sequential' or 'Parallel'.");
            }

            if (normalized.DirectMlDeviceId < -1)
            {
                errors.Add("DirectMlDeviceId must be -1 for automatic / CPU fallback or a valid device id >= 0.");
            }

            return errors;
        }

    }
}
