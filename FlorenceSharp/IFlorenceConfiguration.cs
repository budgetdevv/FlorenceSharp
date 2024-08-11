namespace FlorenceSharp
{
    public interface IFlorenceConfiguration
    {
        public static abstract string EncoderModelPath { get; }
        
        public static abstract string DecoderModelPath { get; }
        
        public static abstract string VisionEncoderModelPath { get; }
        
        public static abstract string TokensEmbeddingModelPath { get; }
    }
}