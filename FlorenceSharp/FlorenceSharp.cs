using System;
using System.Collections.Frozen;
using System.Collections.Generic;
using Microsoft.ML.OnnxRuntime;

namespace FlorenceSharp
{
    public sealed class FlorenceSharp<ConfigT>: IDisposable
        where ConfigT: IFlorenceConfiguration
    {
        // https://huggingface.co/microsoft/Florence-2-large/blob/6bf179230dd8855083a51a5e11beb04aec1291fd/processing_florence2.py#L112
        private static readonly FrozenDictionary<FlorenceMode, string> PROMPTS_WITHOUT_INPUTS = 
            new Dictionary<FlorenceMode, string>() 
            { 
                { FlorenceMode.Caption, "What does this image describe?" },
                { FlorenceMode.DetailedCaption, "Describe in detail what is shown in the image." },
                { FlorenceMode.MoreDetailedCaption, "Describe in detail what is shown in the image." },
                // TODO: Fill in the rest
                
            }.ToFrozenDictionary();

        private const string INPUT_PLACEHOLDER = "{input}";
        
        // https://huggingface.co/microsoft/Florence-2-large/blob/6bf179230dd8855083a51a5e11beb04aec1291fd/processing_florence2.py#L123
        private static readonly FrozenDictionary<FlorenceMode, string> PROMPTS_WITH_INPUTS =
            new Dictionary<FlorenceMode, string>()
            {
                { FlorenceMode.CaptionToPhraseGrounding, $"Locate the phrases in the caption: {INPUT_PLACEHOLDER}" },
                // TODO: Fill in the rest
            }.ToFrozenDictionary();
        
        private readonly SessionOptions OnnxSessionOptions;
        
        private readonly InferenceSession 
            EncoderOnnxSession,
            DecoderOnnxSession,
            VisionEncoderOnnxSession,
            TokensEmbeddingOnnxSession;
        
        public FlorenceSharp(SessionOptions? onnxSessionOptions)
        {
            OnnxSessionOptions = onnxSessionOptions ?? new();
            
            EncoderOnnxSession = new(ConfigT.EncoderModelPath, OnnxSessionOptions);
            DecoderOnnxSession = new(ConfigT.DecoderModelPath, OnnxSessionOptions);
            VisionEncoderOnnxSession = new(ConfigT.VisionEncoderModelPath, OnnxSessionOptions);
            TokensEmbeddingOnnxSession = new(ConfigT.TokensEmbeddingModelPath, OnnxSessionOptions);
        }

        public string GenerateCaption(ReadOnlySpan<byte> imagePixels)
        {
            throw new NotImplementedException();
        }
        
        public string GenerateDetailedCaption(ReadOnlySpan<byte> imagePixels)
        {
            throw new NotImplementedException();
        }
        
        public string GenerateMoreDetailedCaption(ReadOnlySpan<byte> imagePixels)
        {
            throw new NotImplementedException();
        }

        public void Dispose()
        {
            OnnxSessionOptions.Dispose();
            EncoderOnnxSession.Dispose();
            DecoderOnnxSession.Dispose();
            VisionEncoderOnnxSession.Dispose();
            TokensEmbeddingOnnxSession.Dispose();
        }
    }
}