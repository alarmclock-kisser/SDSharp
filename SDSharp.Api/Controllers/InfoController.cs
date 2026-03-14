using Microsoft.AspNetCore.Mvc;
using SDSharp.Shared;
using System.Runtime.CompilerServices;

namespace SDSharp.Api.Controllers
{
    [ApiController]
    [Route("api/[controller]")]
    public class InfoController : ControllerBase
    {
        private readonly AppSettings Settings;


        public InfoController(AppSettings appSettings)
        {
            this.Settings = appSettings;
        }


        [HttpGet("logs")]
        public ActionResult<List<string>> GetLogs()
        {
            try
            {
                var logs = StaticLogger.LogEntriesBindingList.ToList();
                return Ok(logs);
            }
            catch (Exception ex)
            {
                StaticLogger.Log($"Error in GetLogs: {ex.Message}");
                return StatusCode(500, "An error occurred while retrieving logs.");
            }
        }

        [HttpGet("appsettings")]
        public ActionResult<AppSettings> GetAppSettings()
        {
            try
            {
                return Ok(this.Settings);
            }
            catch (Exception ex)
            {
                StaticLogger.Log($"Error in GetAppSettings: {ex.Message}");
                return StatusCode(500, "An error occurred while retrieving app settings.");
            }
        }
    }
}
