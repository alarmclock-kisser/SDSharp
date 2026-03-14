using SDSharp.Shared;
using System;
using System.Collections.Generic;
using System.Text;

namespace SDSharp.Onnx
{
    public partial class OnnxService : IDisposable
    {
        private readonly AppSettings Settings;
        private readonly DxgiHelper Dxgi = new();
        private readonly SemaphoreSlim StateLock = new(1, 1);

        public List<StableDiffusionModel> AvailableModels { get; private set; } = [];
        public List<string> DirectMlDeviceNames { get; private set; } = [];


        public OnnxService(AppSettings appSettings)
        {
            this.Settings = appSettings;

            this.DirectMlDeviceNames = this.Dxgi.GetDirectMlDevices();
            this.AvailableModels = this.GetAvailableModels();
        }



        public List<StableDiffusionModel> GetAvailableModels()
        {
            var modelDirs = this.Settings.ModelDirectories.Distinct().Where(d => Directory.Exists(d));
            var modelRootDirs = modelDirs.SelectMany(d => Directory.GetDirectories(d)).Where(d => Directory.GetFiles(d, "*.onnx", SearchOption.AllDirectories).Any()).ToList();

            try
            {
                return modelRootDirs.Select(d => new StableDiffusionModel(d)).ToList();

            }
            catch (Exception ex)
            {
                StaticLogger.Log($"Error loading models: {ex.Message}");
                return [];
            }
        }




        public void Dispose()
        {
            this.DisposeModelResources();
            this.LoadedModel = null;
            this.StateLock.Dispose();

            GC.SuppressFinalize(this);

        }
    }
}
