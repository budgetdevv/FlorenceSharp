namespace FlorenceSharp.Configs
{
    public interface IDefaultFlorence2Config: IFlorenceConfiguration
    {
        // https://huggingface.co/onnx-community/Florence-2-large/tree/main/onnx

        private const string BASE_PATH = "FlorenceSharp/Models";
        
        static string IFlorenceConfiguration.EncoderModelPath => $"{BASE_PATH}/encoder_model.onnx";
        
        static string IFlorenceConfiguration.DecoderModelPath => $"{BASE_PATH}/decoder_model.onnx";
        
        static string IFlorenceConfiguration.VisionEncoderModelPath => $"{BASE_PATH}/vision_encoder.onnx";
        
        static string IFlorenceConfiguration.TokensEmbeddingModelPath => $"{BASE_PATH}/embed_tokens.onnx";
        
        // Generation
        
        // https://huggingface.co/docs/transformers/en/main_classes/text_generation#transformers.GenerationConfig
        
        // GenerationConfig during python inference: https://imgur.com/a/m8tNdKs
        
        static uint IFlorenceGenerationConfiguration.NoRepeatNgramSize => 3;
        
        static uint IFlorenceGenerationConfiguration.NumBeams => 3;
        
        static uint IFlorenceGenerationConfiguration.MaxLength => 1025;
        
        // The default for GenerationConfig is 50, but seeing that we only care about the top 3 beams,
        // we might want to consider setting this to 3 instead.
        static uint IFlorenceGenerationConfiguration.TopK => 50;
        
        static bool IFlorenceGenerationConfiguration.EarlyStopping => true;
        
        static float IFlorenceGenerationConfiguration.LengthPenalty => 1.0f;
        
        static bool IFlorenceGenerationConfiguration.UseCacheBranch => false;
    }
}