using SixLabors.ImageSharp.Processing;
using SixLabors.ImageSharp.Processing.Processors.Transforms;

namespace FlorenceSharp.Processors.Image
{
    // https://huggingface.co/docs/transformers/en/model_doc/clip#transformers.CLIPImageProcessor
    public interface ICLIPImagePreProcessorConfig
    {
        public static abstract int ImageWidth { get; }
        
        public static abstract int ImageHeight { get; }
        
        public static abstract float[] ImageMean { get; }
        
        public static abstract float[] ImageStd { get; }

        // This is the default, refer to link above
        public static virtual float RescaleFactor => 0.00392156862745098f;
        
        // This is the default, refer to link above
        public static virtual IResampler Resampler => KnownResamplers.Bicubic;
    }
}