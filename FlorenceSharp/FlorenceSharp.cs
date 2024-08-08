using System;
using Microsoft.ML.OnnxRuntime;

namespace FlorenceSharp
{
    public sealed class FlorenceSharp<ConfigT> where ConfigT: IFlorenceConfiguration
    {
        private readonly SessionOptions OnnxSessionOptions;
        
        private readonly InferenceSession OnnxSession;
        
        public FlorenceSharp(SessionOptions? onnxSessionOptions)
        {
            OnnxSessionOptions = onnxSessionOptions ?? new();
            
            var modelPath = ConfigT.ModelPath;
            
            OnnxSession = new(modelPath, OnnxSessionOptions);
        }

        public string GenerateCaption(ReadOnlySpan<byte> imageData)
        {
            throw new NotImplementedException();
        }
        
        public string GenerateDetailedCaption(ReadOnlySpan<byte> imageData)
        {
            throw new NotImplementedException();
        }
        
        public string GenerateMoreDetailedCaption(ReadOnlySpan<byte> imageData)
        {
            throw new NotImplementedException();
        }

        public void Dispose()
        {
            OnnxSessionOptions.Dispose();
            OnnxSession.Dispose();
        }
    }
}