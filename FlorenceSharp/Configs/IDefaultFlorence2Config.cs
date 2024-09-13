namespace FlorenceSharp.Configs
{
    public interface IDefaultFlorence2Config: IFlorenceConfiguration
    {
        // https://huggingface.co/onnx-community/Florence-2-large/tree/main/onnx

        public const string 
            BASE_MODEL_PATH = "FlorenceSharp/Models",
            ENCODER_MODEL_PATH = $"{BASE_MODEL_PATH}/encoder_model.onnx",
            DECODER_MODEL_PATH = $"{BASE_MODEL_PATH}/decoder_model.onnx",
            VISION_ENCODER_MODEL_PATH = $"{BASE_MODEL_PATH}/vision_encoder.onnx",
            TOKENS_EMBEDDING_MODEL_PATH = $"{BASE_MODEL_PATH}/embed_tokens.onnx";

        private static readonly ConfigurableOnnxModel.Configuration
            DEFAULT_ENCODER_MODEL_CONFIG,
            DEFAULT_DECODER_MODEL_CONFIG,
            DEFAULT_VISION_ENCODER_MODEL_CONFIG,
            DEFAULT_TOKENS_EMBEDDING_MODEL_CONFIG;

        static IDefaultFlorence2Config()
        {
            var baseConfig = new ConfigurableOnnxModel.Configuration();
            
            // The config is a struct, so subsequent calls to WithModelPath will not modify the original config.
            DEFAULT_ENCODER_MODEL_CONFIG = baseConfig.WithModelPath(ENCODER_MODEL_PATH);
            DEFAULT_DECODER_MODEL_CONFIG = baseConfig.WithModelPath(DECODER_MODEL_PATH);
            DEFAULT_VISION_ENCODER_MODEL_CONFIG = baseConfig.WithModelPath(VISION_ENCODER_MODEL_PATH);
            DEFAULT_TOKENS_EMBEDDING_MODEL_CONFIG = baseConfig.WithModelPath(TOKENS_EMBEDDING_MODEL_PATH);
        }
        
        static ConfigurableOnnxModel.Configuration IFlorenceConfiguration.EncoderModelConfig
            => DEFAULT_ENCODER_MODEL_CONFIG;
        
        static ConfigurableOnnxModel.Configuration IFlorenceConfiguration.DecoderModelConfig
            => DEFAULT_DECODER_MODEL_CONFIG;
        
        static ConfigurableOnnxModel.Configuration IFlorenceConfiguration.VisionEncoderModelConfig
            => DEFAULT_VISION_ENCODER_MODEL_CONFIG;
        
        static ConfigurableOnnxModel.Configuration IFlorenceConfiguration.TokensEmbeddingModelDeviceType
            => DEFAULT_TOKENS_EMBEDDING_MODEL_CONFIG;
        
        // Generation
        
        // https://huggingface.co/docs/transformers/en/main_classes/text_generation#transformers.GenerationConfig
        
        // GenerationConfig during python inference: https://imgur.com/a/m8tNdKs
        
        // https://huggingface.co/microsoft/Florence-2-large/blob/main/config.json

        
        static uint IFlorenceGenerationConfiguration.NoRepeatNGramSize => 3;
        
        static uint IFlorenceGenerationConfiguration.NumBeams => 3;
        
        static uint IFlorenceGenerationConfiguration.MaxLength => 1025;
        
        // The default for GenerationConfig is 50, but seeing that we only care about the top 3 beams,
        // we might want to consider setting this to 3 instead.
        static uint IFlorenceGenerationConfiguration.TopK => 50;
        
        static bool IFlorenceGenerationConfiguration.EarlyStopping => true;
        
        static float IFlorenceGenerationConfiguration.LengthPenalty => 1.0f;
        
        static uint IFlorenceGenerationConfiguration.EncoderAttentionHeads => 16;
        
        static uint IFlorenceGenerationConfiguration.EncoderLayers => 12;
        
        static uint IFlorenceGenerationConfiguration.DecoderAttentionHeads => 16;
        
        static uint IFlorenceGenerationConfiguration.DecoderLayers => 12;
    }
}