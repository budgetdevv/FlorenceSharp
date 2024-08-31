namespace FlorenceSharp.Configs
{
    public interface IFlorenceConfiguration: IFlorenceGenerationConfiguration
    {
        public static abstract string EncoderModelPath { get; }
        
        public static abstract string DecoderModelPath { get; }
        
        public static abstract string VisionEncoderModelPath { get; }
        
        public static abstract string TokensEmbeddingModelPath { get; }
    }

    public interface IFlorenceGenerationConfiguration
    {
        public static abstract uint NoRepeatNgramSize { get; }
        
        public static abstract uint NumBeams { get; }
        
        public static abstract uint MaxLength { get; }
        
        public static abstract uint TopK { get; }
        
        public static abstract bool EarlyStopping { get; }
        
        public static abstract float LengthPenalty { get; }
    }
}