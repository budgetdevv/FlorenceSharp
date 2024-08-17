using System;
using System.Reflection;
using FlorenceSharp.Helpers;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace FlorenceSharp.Tokenizers
{
    public readonly struct FlorenceBartTokenizer: IDisposable
    {
        private static readonly Assembly CURRENT_ASSEMBLY = typeof(FlorenceBartTokenizer).Assembly;
        
        public struct EncoderInput
        {
            public const string TEXT_INPUT_NAME = "input_text";
        }
        
        public struct EncoderOutput(DenseTensor<long> inputIDs, DenseTensor<long> attentionMask)
        {
            public DenseTensor<long> InputIDs = inputIDs;
            
            public DenseTensor<long> AttentionMask = attentionMask;
        }

        public struct DecoderInput
        {
            public const string IDS_INPUT_NAME = "ids";
        }
         
        // public struct DecoderOutput
        // {
        //     public const string TEXT_OUTPUT_NAME = "str";
        // }
        
        // I wanted to use Microsoft.ML.Tokenizers, but unfortunately there isn't a straightforward way to add special tokens
        // See: https://github.com/dotnet/machinelearning/issues/6901

        public readonly SessionOptions SessionOptions;
        
        private readonly InferenceSession TokenizerEncodeSession, TokenizerDecodeSession;

        private const string 
            ENCODER_MODEL_PATH = "florence2_tokenizer_encode.onnx",
            DECODER_MODEL_PATH = "florence2_tokenizer_decode.onnx";
        
        public FlorenceBartTokenizer(SessionOptions sessionOptions)
        {
            sessionOptions.RegisterOrtExtensions();
            SessionOptions = sessionOptions;

            var currentAssembly = CURRENT_ASSEMBLY;
            
            TokenizerEncodeSession = new(
                ResourceHelpers.GetResourceBytes(currentAssembly, ENCODER_MODEL_PATH)!,
                sessionOptions);
            
            TokenizerDecodeSession = new(
                ResourceHelpers.GetResourceBytes(currentAssembly, DECODER_MODEL_PATH)!,
                sessionOptions);
        }

        public EncoderOutput Tokenize(string sentence)
        {
            string[] arr = [ sentence ];
            return Tokenize(arr);
        }
        
        public EncoderOutput Tokenize(Memory<string> sentences)
        {
            var inputTensor = new DenseTensor<string>(sentences, [ sentences.Length ]);
            
            var output = TokenizerEncodeSession.Run(
            [
                NamedOnnxValue.CreateFromTensor(EncoderInput.TEXT_INPUT_NAME, inputTensor),
            ]);

            var inputIDs = (DenseTensor<long>) output[0].Value;
            var attentionMask = (DenseTensor<long>) output[1].Value;
            
            return new(inputIDs, attentionMask);
        }

        public string Decode(Memory<long> inputIDs)
        {
            var inputTensor = new DenseTensor<long>(inputIDs, [ inputIDs.Length ]);
            
            var output = TokenizerDecodeSession.Run(
            [
                NamedOnnxValue.CreateFromTensor(DecoderInput.IDS_INPUT_NAME, inputTensor),
            ]);
            
            var outputTensor = (DenseTensor<string>) output[0].Value;

            // Get the string from the tensor ( There's only 1 string )
            return outputTensor[0];
        }
        
        public void Dispose()
        {
            TokenizerEncodeSession.Dispose();
            TokenizerDecodeSession.Dispose();
        }
    }
}