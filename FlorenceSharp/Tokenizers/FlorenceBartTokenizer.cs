using System;
using System.Collections.Frozen;
using System.Collections.Generic;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text.Json;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using ONNX.Common.Helpers;

namespace FlorenceSharp.Tokenizers
{
    public readonly struct FlorenceBartTokenizer: IDisposable
    {
        public struct EncoderInput
        {
            public const string TEXT_INPUT_NAME = "input_text";
        }
        
        public readonly struct EncoderOutput(
            IDisposableReadOnlyCollection<DisposableNamedOnnxValue> outputs,
            DenseTensor<long> inputIDs,
            DenseTensor<long> attentionMask): IDisposable
        {
            private readonly IDisposableReadOnlyCollection<DisposableNamedOnnxValue> Outputs = outputs;
            
            public readonly DenseTensor<long> InputIDs = inputIDs;
            
            public readonly DenseTensor<long> AttentionMask = attentionMask;
            
            public void Dispose()
            {
                Outputs.Dispose();
            }
        }

        public struct DecoderInput
        {
            public const string IDS_INPUT_NAME = "ids";
        }
        
        public struct DecoderOutput
        {
            public const string DECODED_TEXT_NAME = "str";
        }
        
        private const string 
            ENCODER_MODEL_PATH = "florence2_tokenizer_encode.onnx",
            DECODER_MODEL_PATH = "florence2_tokenizer_decode.onnx",
            DECODER_SKIP_SPECIAL_TOKENS_MODEL_PATH = "florence2_tokenizer_decode_skip_special_tokens.onnx",
            VOCAB_PATH = "vocab.json";

        private static readonly Assembly CURRENT_ASSEMBLY = typeof(FlorenceBartTokenizer).Assembly;
        
        
        public readonly SessionOptions SessionOptions;
        
        private readonly InferenceSession 
            TokenizerEncodeSession,
            TokenizerDecodeSession,
            TokenizerDecodeSkipSpecialTokensSession;

        public readonly FrozenDictionary<string, int> VocabularyToTokenIDMap;
        
        public readonly string[] Vocabulary;
        
        public FlorenceBartTokenizer()
        {
            throw new NotSupportedException();
        }
        
        public FlorenceBartTokenizer(SessionOptions sessionOptions)
        {
            SessionOptions = sessionOptions;
            
            sessionOptions.RegisterOrtExtensions();

            var currentAssembly = CURRENT_ASSEMBLY;
            
            TokenizerEncodeSession = new(
                ResourceHelpers.GetResourceBytes(currentAssembly, ENCODER_MODEL_PATH)!,
                sessionOptions);
            
            TokenizerDecodeSession = new(
                ResourceHelpers.GetResourceBytes(currentAssembly, DECODER_MODEL_PATH)!,
                sessionOptions);
            
            TokenizerDecodeSkipSpecialTokensSession = new(
                ResourceHelpers.GetResourceBytes(currentAssembly, DECODER_SKIP_SPECIAL_TOKENS_MODEL_PATH)!,
                sessionOptions);

            var map = VocabularyToTokenIDMap = JsonSerializer
                .Deserialize<Dictionary<string, int>>(
                    ResourceHelpers.GetResourceBytes(currentAssembly, VOCAB_PATH))!
                .ToFrozenDictionary();
            
            Vocabulary = ImmutableCollectionsMarshal.AsArray(map.Keys)!;
        }

        public EncoderOutput Tokenize(string sentence)
        {
            string[] arr = [ sentence ];
            return Tokenize(arr);
        }
        
        public EncoderOutput Tokenize(Memory<string> sentences)
        {
            var inputTensor = new DenseTensor<string>(sentences, [ sentences.Length ]);
         
            // https://imgur.com/a/LJacvsC
            var output = TokenizerEncodeSession.Run(
            [
                NamedOnnxValue.CreateFromTensor(EncoderInput.TEXT_INPUT_NAME, inputTensor),
            ]);
            
            var inputIDs = (DenseTensor<long>) output[0].Value;
            var attentionMask = (DenseTensor<long>) output[1].Value;
            
            // Caller will be responsible for disposing the output
            return new(output, inputIDs, attentionMask);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public string Decode(Memory<long> inputIDs, bool skipSpecialTokens)
        {
            var inputTensor = new DenseTensor<long>(inputIDs, [ inputIDs.Length ]);
            
            var session = skipSpecialTokens ? TokenizerDecodeSkipSpecialTokensSession : TokenizerDecodeSession;
         
            // https://imgur.com/a/gHk84M6
            var output = session.Run(
                inputs: 
                [
                    NamedOnnxValue.CreateFromTensor(DecoderInput.IDS_INPUT_NAME, inputTensor),
                ]
            );
            
            var outputTensor = (DenseTensor<string>) output[0].Value;

            // Get the string from the tensor ( There's only 1 string )
            return outputTensor[0];
        }
        
        public int GetTokenID(string token)
        {
            // Yes, we actually want it to throw an exception if the token doesn't exist
            return VocabularyToTokenIDMap[token];
        }
        
        public void Dispose()
        {
            TokenizerEncodeSession.Dispose();
            TokenizerDecodeSession.Dispose();
            SessionOptions.Dispose();
        }
    }
}