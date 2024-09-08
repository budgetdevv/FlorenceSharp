namespace FlorenceSharp.Configs
{
    public interface IFlorenceConfiguration: IFlorenceGenerationConfiguration
    {
        public static abstract ConfigurableOnnxModel.Configuration EncoderModelConfig { get; }
        
        public static abstract ConfigurableOnnxModel.Configuration DecoderModelConfig { get; }
        
        public static abstract ConfigurableOnnxModel.Configuration VisionEncoderModelConfig { get; }
        
        public static abstract ConfigurableOnnxModel.Configuration TokensEmbeddingModelDeviceType { get; }
    }

    public interface IFlorenceGenerationConfiguration
    {
        public static abstract uint NoRepeatNGramSize { get; }
        
        public static abstract uint NumBeams { get; }
        
        public static abstract uint MaxLength { get; }
        
        public static abstract uint TopK { get; }
        
        public static abstract bool EarlyStopping { get; }
        
        public static abstract float LengthPenalty { get; }
        
        public static abstract bool UseCacheBranch { get; }

        public static abstract uint EncoderAttentionHeads { get; }
        
        public static abstract uint EncoderLayers { get; }
        
        public static abstract uint DecoderAttentionHeads { get; }
        
        public static abstract uint DecoderLayers { get; }
    }
}