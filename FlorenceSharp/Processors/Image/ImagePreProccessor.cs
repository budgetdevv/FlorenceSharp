using System;

namespace FlorenceSharp.Processors.Image
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
            throw new NotImplementedException();
        }
    }
}