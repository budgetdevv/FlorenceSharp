using System;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace FlorenceSharp.Processors.Imaging
{
    public readonly struct CLIPImagePreProcessor<ConfigT> where ConfigT: ICLIPImagePreProcessorConfig
    {
        public struct Output(float[] normalizedInput, ImageDimensions originalDimensions)
        {
            public float[] NormalizedInput = normalizedInput;
            
            public ImageDimensions OriginalDimensions = originalDimensions;
        }
        
        public Output PreProcess(ReadOnlySpan<byte> imagePixels)
        {
            var imageWidth = ConfigT.ImageWidth;
            var imageHeight = ConfigT.ImageHeight;
            
            var buffer = new float[imageWidth * imageHeight * 3];
            
            using var image = Image.Load<Rgba32>(imagePixels);

            var originalDimensions = new ImageDimensions(image.Width, image.Height);
            
            image.Mutate(x => x.Resize(imageWidth, imageHeight));
            
            image.ProcessPixelRows(pixelAccessor =>
            {
                var imageMean = ConfigT.ImageMean;
                var imageMean0 = imageMean[0];
                var imageMean1 = imageMean[1];
                var imageMean2 = imageMean[2];
                
                var imageStd = ConfigT.ImageStd;
                var imageStd0 = imageStd[0];
                var imageStd1 = imageStd[1];
                var imageStd2 = imageStd[2];
                
                for (var y = 0; y < imageHeight; y++)
                {
                    var rowSpan = pixelAccessor.GetRowSpan(y);
                
                    for (var x = 0; x < imageWidth; x++)
                    {
                        var pixel = rowSpan[x];

                        var offset = (y * imageWidth + x) * 3;
                    
                        buffer[offset + 0] = (pixel.R - imageMean0) / imageStd0;
                        buffer[offset + 1] = (pixel.G - imageMean1) / imageStd1;
                        buffer[offset + 2] = (pixel.B - imageMean2) / imageStd2;
                    }
                }
            });
            
            return new(buffer, originalDimensions);
        }
    }
}