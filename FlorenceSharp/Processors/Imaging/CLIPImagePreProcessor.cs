using System;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace FlorenceSharp.Processors.Imaging
{
    public readonly struct CLIPImagePreProcessor<ConfigT> where ConfigT: ICLIPImagePreProcessorConfig
    {
        public readonly struct Output(
            DenseTensor<float> normalizedInputTensor,
            int[] tensorDimensions,
            ImageDimensions imageOriginalDimensions)
        {
            public readonly DenseTensor<float> NormalizedInputTensor = normalizedInputTensor;
            
            public readonly int[] TensorDimensions = tensorDimensions;
            
            public readonly ImageDimensions ImageOriginalDimensions = imageOriginalDimensions;
        }
        
        public Output PreProcess(ReadOnlySpan<byte> imagePixels)
        {
            var imageWidth = ConfigT.ImageWidth;
            var imageHeight = ConfigT.ImageHeight;
            
            int[] dimensions = [ 3, imageWidth, imageHeight ];
            
            var tensor = new DenseTensor<float>(dimensions);
            
            using var image = Image.Load<Rgba32>(imagePixels);

            var originalImageDimensions = new ImageDimensions(image.Width, image.Height);
            
            image.Mutate(x => x.Resize(
                imageWidth, 
                imageHeight,
                ConfigT.Resampler, 
                compand: false));
            
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
                
                var rescaleFactor = ConfigT.RescaleFactor;
                
                for (var y = 0; y < imageHeight; y++)
                {
                    var rowSpan = pixelAccessor.GetRowSpan(y);
                
                    for (var x = 0; x < imageWidth; x++)
                    {
                        var pixel = rowSpan[x];
                        
                        tensor[0, y, x] = ((pixel.R * rescaleFactor) - imageMean0) / imageStd0;
                        tensor[1, y, x] = ((pixel.G * rescaleFactor) - imageMean1) / imageStd1;
                        tensor[2, y, x] = ((pixel.B * rescaleFactor) - imageMean2) / imageStd2;
                    }
                }
            });
            
            return new(tensor, dimensions, originalImageDimensions);
        }
    }
}