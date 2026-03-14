
using SDSharp.Onnx;
using SDSharp.Shared;

namespace SDSharp.Api
{
    public class Program
    {
        public static void Main(string[] args)
        {
            var builder = WebApplication.CreateBuilder(args);

            // Get appsettings (+DTO)
            var appSettings = builder.Configuration.GetSection("AppSettings").Get<AppSettings>() ?? new();
            builder.Services.AddSingleton(appSettings);
            bool createLogFile = builder.Configuration.GetValue<bool>("CreateLogFile");
            string logDirectory = builder.Configuration.GetValue<string>("LogDirectory") ?? Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Logs");
            int maxLogFiles = builder.Configuration.GetValue<int>("MaxLogFiles", 32);

            // Initialize StaticLogger
            StaticLogger.InitializeLogFiles(logDirectory, createLogFile, maxLogFiles);
            StaticLogger.SetUiContext(SynchronizationContext.Current ?? new SynchronizationContext());

            // Add services to the container.
            builder.Services.AddSingleton<OnnxService>(sp => new OnnxService(appSettings));


            builder.Services.AddControllers();
            // Swagger / OpenAPI
            builder.Services.AddEndpointsApiExplorer();
            builder.Services.AddSwaggerGen();

            var app = builder.Build();

            // Configure the HTTP request pipeline.
            if (app.Environment.IsDevelopment())
            {
                app.UseSwagger();
                app.UseSwaggerUI(c => c.SwaggerEndpoint("/swagger/v1/swagger.json", "SDSharp API V1"));
            }

            app.UseHttpsRedirection();

            app.UseAuthorization();


            app.MapControllers();

            app.Run();
        }
    }
}
