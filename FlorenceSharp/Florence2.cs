using System;
using System.Collections.Frozen;
using System.Collections.Generic;
using System.Threading.Tasks;
using FlorenceSharp.Processors.Imaging;
using FlorenceSharp.Tokenizers;
using Microsoft.ML.OnnxRuntime;

namespace FlorenceSharp
{
    public struct DefaultFlorence2Config: IFlorenceConfiguration
    {
        // https://huggingface.co/onnx-community/Florence-2-large/tree/main/onnx

        private const string BASE_PATH = "FlorenceSharp/Models";
        
        public static string EncoderModelPath => $"{BASE_PATH}/encoder_model.onnx";
        
        public static string DecoderModelPath => $"{BASE_PATH}/decoder_model.onnx";
        
        public static string VisionEncoderModelPath => $"{BASE_PATH}/vision_encoder.onnx";
        
        public static string TokensEmbeddingModelPath => $"{BASE_PATH}/embed_tokens.onnx";
    }

    public sealed class Florence2(SessionOptions? onnxSessionOptions = null):
        Florence2<DefaultFlorence2Config>(onnxSessionOptions);
    
    public class Florence2<ConfigT>: IAsyncInitializable<SessionOptions?, Florence2<ConfigT>>, IDisposable
        where ConfigT: struct, IFlorenceConfiguration
    {
        // https://huggingface.co/microsoft/Florence-2-large/blob/6bf179230dd8855083a51a5e11beb04aec1291fd/processing_florence2.py#L112
        private static readonly FrozenDictionary<FlorenceMode, string> PROMPTS_WITHOUT_INPUTS = 
            new Dictionary<FlorenceMode, string>() 
            { 
                { FlorenceMode.Caption, "What does this image describe?" },
                { FlorenceMode.DetailedCaption, "Describe in detail what is shown in the image." },
                { FlorenceMode.MoreDetailedCaption, "Describe with a paragraph what is shown in the image." },
                { FlorenceMode.OCR, "What is the text in the image?" },
                { FlorenceMode.OCRWithRegion, "What is the text in the image, with regions?" },
                { FlorenceMode.ObjectDetection, "Locate the objects with category name in the image." },
                { FlorenceMode.DenseRegionCaption, "Locate the objects in the image, with their descriptions." },
                { FlorenceMode.RegionProposal, "Locate the region proposals in the image." },
                
            }.ToFrozenDictionary();

        private const string INPUT_PLACEHOLDER = "{input}";
        
        // https://huggingface.co/microsoft/Florence-2-large/blob/6bf179230dd8855083a51a5e11beb04aec1291fd/processing_florence2.py#L123
        private static readonly FrozenDictionary<FlorenceMode, string> PROMPTS_WITH_INPUTS =
            new Dictionary<FlorenceMode, string>()
            {
                { FlorenceMode.CaptionToPhraseGrounding, $"Locate the phrases in the caption: {INPUT_PLACEHOLDER}" },
                { FlorenceMode.ReferringExpressionSegmentation, $"Locate {INPUT_PLACEHOLDER} in the image with mask" },
                { FlorenceMode.RegionToSegmentation, $"What is the polygon mask of region {INPUT_PLACEHOLDER}" },
                { FlorenceMode.OpenVocabularyDetection, $"Locate {INPUT_PLACEHOLDER} in the image." },
                { FlorenceMode.RegionToCategory, $"What is the region {INPUT_PLACEHOLDER}?" },
                { FlorenceMode.RegionToDescription, $"What does the region {INPUT_PLACEHOLDER} describe?" },
                { FlorenceMode.RegionToOCR, $"What text is in the region {INPUT_PLACEHOLDER}?" },
            }.ToFrozenDictionary();
        
        private readonly SessionOptions OnnxSessionOptions;
        
        private readonly InferenceSession 
            EncoderOnnxSession,
            DecoderOnnxSession,
            VisionEncoderOnnxSession,
            TokensEmbeddingOnnxSession;
        
        // https://huggingface.co/microsoft/Florence-2-large/blob/main/preprocessor_config.json
        private struct CLIPImageProcessorConfig: ICLIPImagePreProcessorConfig
        {
            public static int ImageWidth => 768;
            
            public static int ImageHeight => ImageWidth;
            
            public static int ImageSequenceLength => 577;

            public static float[] ImageMean => [ 0.485f, 0.456f, 0.406f ];
            
            public static float[] ImageStd => [ 0.229f, 0.224f, 0.225f ];
        }

        private readonly CLIPImagePreProcessor<CLIPImageProcessorConfig> ImagePreProcessor;
        
        private readonly FlorenceBartTokenizer Tokenizer;

        protected Florence2(SessionOptions? onnxSessionOptions = null)
        {
            OnnxSessionOptions = onnxSessionOptions ??= new();
            
            EncoderOnnxSession = new(ConfigT.EncoderModelPath, OnnxSessionOptions);
            DecoderOnnxSession = new(ConfigT.DecoderModelPath, OnnxSessionOptions);
            VisionEncoderOnnxSession = new(ConfigT.VisionEncoderModelPath, OnnxSessionOptions);
            TokensEmbeddingOnnxSession = new(ConfigT.TokensEmbeddingModelPath, OnnxSessionOptions);
            
            ImagePreProcessor = new();
            
            Tokenizer = new(onnxSessionOptions);
        }
        
        public static async ValueTask<Florence2<ConfigT>> InitializeAsync(SessionOptions? onnxSessionOptions)
        {
            return new(onnxSessionOptions);
        }
        
        public string GenerateCaption(ReadOnlySpan<byte> imagePixels)
        {
            return GenerateCaptionCore(imagePixels, FlorenceMode.Caption);
        }
        
        public string GenerateDetailedCaption(ReadOnlySpan<byte> imagePixels)
        {
            return GenerateCaptionCore(imagePixels, FlorenceMode.DetailedCaption);
        }
        
        public string GenerateMoreDetailedCaption(ReadOnlySpan<byte> imagePixels)
        {
            return GenerateCaptionCore(imagePixels, FlorenceMode.MoreDetailedCaption);
        }

        private string GenerateCaptionCore(ReadOnlySpan<byte> imagePixels, FlorenceMode mode)
        {
            var prompt = PROMPTS_WITHOUT_INPUTS[mode];
            
            var encoded = Tokenizer.Tokenize(prompt);
            
            var imagePreProcessed = ImagePreProcessor.PreProcess(imagePixels);
            
            
            
            throw new NotImplementedException();
        }
        
        public void Dispose()
        {
            OnnxSessionOptions.Dispose();
            
            EncoderOnnxSession.Dispose();
            DecoderOnnxSession.Dispose();
            VisionEncoderOnnxSession.Dispose();
            TokensEmbeddingOnnxSession.Dispose();
            
            Tokenizer.Dispose();
        }
    }
}