using Microsoft.AspNetCore.Mvc;
using SDSharp.Onnx;
using SDSharp.Shared;
using System.Runtime.Versioning;

namespace SDSharp.Api.Controllers
{
    [ApiController]
    [Route("api/[controller]")]
    public class StableDiffusionController : ControllerBase
    {
        private readonly OnnxService Onnx;

        public StableDiffusionController(OnnxService onnxService)
        {
            this.Onnx = onnxService;
        }

        [HttpGet("models")]
        public ActionResult<List<StableDiffusionModel>> GetAvailableModels()
        {
            try
            {
                return Ok(this.Onnx.AvailableModels);
            }
            catch (Exception ex)
            {
                StaticLogger.Log($"Error in GetAvailableModels: {ex.Message}");
                return StatusCode(500, "An error occurred while retrieving available models.");
            }
        }

        [HttpGet("directml-devices")]
        public ActionResult<List<string>> GetDirectMlDevices()
        {
            try
            {
                return Ok(this.Onnx.DirectMlDeviceNames);
            }
            catch (Exception ex)
            {
                StaticLogger.Log($"Error in GetDirectMlDevices: {ex.Message}");
                return StatusCode(500, "An error occurred while retrieving DirectML devices.");
            }
        }

        [HttpGet("loaded")]
        public ActionResult<StableDiffusionModel?> GetLoadedModel()
        {
            return Ok(this.Onnx.LoadedModel);
        }

        [HttpPost("load")]
        public async Task<ActionResult<StableDiffusionModel?>> LoadModelAsync([FromBody] StableDiffusionLoadRequest request, CancellationToken ct)
        {
            try
            {
                var errors = request.Validate();
                if (errors.Count > 0)
                {
                    return BadRequest(errors);
                }

                var model = await this.Onnx.LoadModelAsync(request.Model, request.Options, this.CreateLoggingProgress(request.LogProgress, "load"), ct);
                if (model == null)
                {
                    return BadRequest("The model could not be loaded. Check the API logs for details.");
                }

                return Ok(model);
            }
            catch (OperationCanceledException)
            {
                return StatusCode(499);
            }
            catch (Exception ex)
            {
                await StaticLogger.LogAsync(ex, "Error in LoadModelAsync");
                return StatusCode(500, "An error occurred while loading the model.");
            }
        }

        [HttpPost("unload")]
        public async Task<ActionResult<bool?>> UnloadModelAsync(CancellationToken ct)
        {
            try
            {
                ct.ThrowIfCancellationRequested();
                return Ok(await this.Onnx.UnloadModelAsync());
            }
            catch (OperationCanceledException)
            {
                return StatusCode(499);
            }
            catch (Exception ex)
            {
                await StaticLogger.LogAsync(ex, "Error in UnloadModelAsync");
                return StatusCode(500, "An error occurred while unloading the model.");
            }
        }

        [SupportedOSPlatform("windows")]
        [HttpPost("generate")]
        public async Task<ActionResult<ImageObj>> GenerateAsync([FromBody] StableDiffusionGenerateRequest request, CancellationToken ct)
        {
            try
            {
                var errors = request.Validate();
                if (errors.Count > 0)
                {
                    return BadRequest(errors);
                }

                var image = await this.Onnx.GenerateImageAsync(request, this.CreateLoggingProgress(request.LogProgress, "generate"), ct);
                if (image == null)
                {
                    return BadRequest("The image could not be generated. Check the API logs for details.");
                }

                return Ok(image);
            }
            catch (OperationCanceledException)
            {
                return StatusCode(499);
            }
            catch (Exception ex)
            {
                await StaticLogger.LogAsync(ex, "Error in GenerateAsync");
                return StatusCode(500, "An error occurred while generating the image.");
            }
        }

        [SupportedOSPlatform("windows")]
        [HttpPost("generate/file")]
        public async Task<IActionResult> GenerateFileAsync([FromBody] StableDiffusionGenerateRequest request, CancellationToken ct)
        {
            try
            {
                var errors = request.Validate();
                if (errors.Count > 0)
                {
                    return BadRequest(errors);
                }

                var image = await this.Onnx.GenerateImageAsync(request, this.CreateLoggingProgress(request.LogProgress, "generate-file"), ct);
                if (image == null)
                {
                    return BadRequest("The image could not be generated. Check the API logs for details.");
                }

                return File(image.Data, image.MediaType, image.FileName);
            }
            catch (OperationCanceledException)
            {
                return StatusCode(499);
            }
            catch (Exception ex)
            {
                await StaticLogger.LogAsync(ex, "Error in GenerateFileAsync");
                return StatusCode(500, "An error occurred while generating the image file.");
            }
        }

        private IProgress<double>? CreateLoggingProgress(bool logProgress, string operation)
        {
            if (!logProgress)
            {
                return null;
            }

            return new Progress<double>(value =>
            {
                double clamped = Math.Clamp(value, 0.0, 1.0);
                StaticLogger.Log($"StableDiffusion {operation} progress: {(clamped*100):F2}%");
            });
        }
    }
}
