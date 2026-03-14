using SDSharp.Shared;
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO.Compression;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SharpAI.Core
{
    public class ImageCollection
    {
        public readonly ConcurrentDictionary<Guid, ImageObj> Images = [];

        public int Count => this.Images.Count;

        public ImageObj? this[Guid id] => this.Images.TryGetValue(id, out var imgObj) ? imgObj : null;

        public ImageObj? this[string nameOrGuid]
        {
            get
            {
                // Try by Guid first
                if (Guid.TryParse(nameOrGuid, out var guid))
                {
                    return this.Images.TryGetValue(guid, out var imgObj) ? imgObj : null;
                }
                // Then by name (contains)
                return this.Images.Values.FirstOrDefault(img => img.FilePath.IndexOf(nameOrGuid, StringComparison.OrdinalIgnoreCase) >= 0);
            }
        }

        public ImageObj? this[int index]
        {
            get
            {
                if (index < 0 || index >= this.Images.Count)
                {
                    return null;
                }
                return this.Images.Values.ElementAt(index);
            }
        }



        public ImageCollection(bool loadEmbeddedResources = false, string[]? additionalRessourcePaths = null)
        {
            if (loadEmbeddedResources)
            {
                // Force enumeration so resources are loaded into the collection
                _ = this.LoadResourcesImages().ToList();
            }

            if (additionalRessourcePaths != null)
            {
                // Load additional resource images from specified file paths or directory get all images from directory
                foreach (var path in additionalRessourcePaths)
                {
                    if (System.IO.Directory.Exists(path))
                    {
                        var files = System.IO.Directory.GetFiles(path, "*.*", System.IO.SearchOption.AllDirectories)
                            .Where(file => file.EndsWith(".jpg", StringComparison.OrdinalIgnoreCase) ||
                                           file.EndsWith(".jpeg", StringComparison.OrdinalIgnoreCase) ||
                                           file.EndsWith(".png", StringComparison.OrdinalIgnoreCase) ||
                                           file.EndsWith(".bmp", StringComparison.OrdinalIgnoreCase));
                        foreach (var file in files)
                        {
                            this.ImportImage(file);
                        }
                    }
                    else if (System.IO.File.Exists(path))
                    {
                        this.ImportImage(path);
                    }
                }
            }
        }



        public bool AddImage(ImageObj imgObj, bool disposeWhenNotAdded = true)
        {
            if (imgObj == null)
            {
                return false;
            }

            var added = this.Images.TryAdd(imgObj.Id, imgObj);
            if (!added && disposeWhenNotAdded)
            {
                StaticLogger.Log($"Disposing image Id: {imgObj.Id} as it was not added to collection");
                imgObj.Dispose();
            }

            StaticLogger.Log(added
                ? $"Added image Id: {imgObj.Id}"
                : $"Image Id: {imgObj.Id} already exists in collection");
            return added;
        }

        public ImageObj? ImportImage(string filePath, bool disposeWhenNotAdded = true)
        {
            try
            {
                var imgObj = new ImageObj(filePath);
                StaticLogger.Log($"Imported image from file: {filePath} as Id: {imgObj.Id}");
                return this.AddImage(imgObj, disposeWhenNotAdded) ? imgObj : null;
            }
            catch
            {
                return null;
            }
        }

        public async Task<ImageObj?> ImportImageAsync(string filePath, bool disposeWhenNotAdded = true)
        {
            return await Task.Run(() => this.ImportImage(filePath, disposeWhenNotAdded));
        }

        public IEnumerable<ImageObj> LoadResourcesImages()
        {
            // Load embedded resource images (jpg, png, bmp) from the assembly
            var assembly = typeof(ImageCollection).Assembly;
            var resourceNames = assembly.GetManifestResourceNames()
                .Where(name => name.EndsWith(".jpg", StringComparison.OrdinalIgnoreCase) ||
                               name.EndsWith(".png", StringComparison.OrdinalIgnoreCase) ||
                               name.EndsWith(".bmp", StringComparison.OrdinalIgnoreCase));
            foreach (var resourceName in resourceNames)
            {
                using var stream = assembly.GetManifestResourceStream(resourceName);
                if (stream != null)
                {
                    var imgObj = new ImageObj(stream);
                    if (this.AddImage(imgObj))
                    {
                        StaticLogger.Log($"Loaded embedded resource image: {resourceName} as Id: {imgObj.Id}");
                        yield return imgObj;
                    }
                    else
                    {
                        imgObj.Dispose();
                    }
                }
            }
        }

        public bool RemoveImage(Guid imgId, bool dispose = true)
        {
            if (this.Images.TryRemove(imgId, out var imgObj))
            {
                if (dispose)
                {
                    StaticLogger.Log($"Disposing image Id: {imgId}");
                    imgObj.Dispose();
                }
                return true;
            }

            return false;
        }

        public void Clear(bool dispose = true)
        {
            foreach (var kvp in this.Images.ToArray())
            {
                if (this.Images.TryRemove(kvp.Key, out var imgObj) && dispose)
                {
                    StaticLogger.Log($"Disposing image Id: {kvp.Key}");
                    imgObj.Dispose();
                }
            }
        }





        public async Task<string> GenerateThumbnailFromBase64Async(string base64ImageData, int pxDiagonal)
        {
            var imgObj = new ImageObj("image/png", base64ImageData);
            var thumbnailBase64 = await imgObj.GetThumbnailBase64Async(pxDiagonal);
            return thumbnailBase64 ?? string.Empty;
        }


        public static async Task<string?> ExportZipAsync(IEnumerable<ImageObj> imageObjs, string? exportPath = null, string format = "png")
        {
            // Verify format
            format = format.ToLower();
            if (format != "png" && format != "jpg" && format != "jpeg" && format != "bmp")
            {
                format = "png";
                await StaticLogger.LogAsync($"Unsupported format specified for export: {format}. Defaulting to PNG.");
            }

            // Verify export path
            exportPath ??= Path.Combine(Path.GetTempPath(), $"ImageCollectionExport_{DateTime.Now:yyyyMMdd_HHmmss}.zip");

            // Async and parallel export of images to temporary files
            var tempFiles = new ConcurrentBag<string>();
            var exportTasks = imageObjs.Select(async imgObj =>
            {
                try
                {
                    string tempFile = Path.Combine(Path.GetTempPath(), $"{imgObj.Id}.{format}");
                    var exportFile = await imgObj.ExportAsync(tempFile, format);
                    if (exportFile != null)
                    {
                        tempFiles.Add(exportFile);
                    }
                }
                catch (Exception ex)
                {
                    await StaticLogger.LogAsync($"Error exporting image Id: {imgObj.Id} - {ex.Message}");
                }
            });

            await Task.WhenAll(exportTasks);

            // Create ZIP archive
            var zipFile = new FileInfo(exportPath);
            try
            {
                using (var zip = new System.IO.Compression.ZipArchive(zipFile.OpenWrite(), System.IO.Compression.ZipArchiveMode.Create))
                {
                    foreach (var tempFile in tempFiles)
                    {
                        var entryName = Path.GetFileName(tempFile);
                        zip.CreateEntryFromFile(tempFile, entryName);
                    }
                }
                await StaticLogger.LogAsync($"Exported {tempFiles.Count} images to ZIP: {exportPath}");
                return exportPath;
            }
            catch (Exception ex)
            {
                await StaticLogger.LogAsync($"Error creating ZIP archive: {ex.Message}");
                return null;
            }
            finally
            {
                // Clean up temporary files
                foreach (var tempFile in tempFiles)
                {
                    try
                    {
                        if (File.Exists(tempFile))
                        {
                            File.Delete(tempFile);
                            await StaticLogger.LogAsync($"Deleted temporary file: {tempFile}");
                        }
                    }
                    catch (Exception ex)
                    {
                        await StaticLogger.LogAsync($"Error deleting temporary file: {tempFile} - {ex.Message}");
                    }
                }
            }

        }




    }
}
