using System;
using FlorenceSharp.Configs;
using FlorenceSharp.Helpers;
using FlorenceSharp.Tensor;
using FlorenceSharp.Tokenizers;

namespace FlorenceSharp.DecodingStrategies
{
    public struct BeamSearch<ConfigT> where ConfigT: struct, IFlorenceGenerationConfiguration
    {
        private readonly struct SampleResult(string token, double score)
        {
            public readonly string Token = token;
            
            public readonly double Score = score;
        }
        
        public uint NumBeams => ConfigT.NumBeams;
        
        private readonly SampleResult[] ResultsBuffer;
        
        public BeamSearch()
        {
            ResultsBuffer = new SampleResult[NumBeams];
        }

        // Technically could turn this into ref struct enumerator and avoid re-iterating.
        public void Sample(ManagedTensor<float> logits, in FlorenceBartTokenizer tokenizer)
        {
            // Get the TopK results
            
            var topKResult = logits.TopK(ConfigT.TopK);

            var logitsSoftmax = topKResult.Logits.SoftMaxInPlace().ValuesArr;
            
            var topKIndices = topKResult.Indices.ValuesArr;
            
            var numBeams = ResultsBuffer.Length;

            var vocab = tokenizer.Vocabulary;
            
            for (int i = 0; i < numBeams; i++)
            {
                var token = vocab[topKIndices[i]];
                
                ResultsBuffer[i] = new(
                    token: token,
                    score: Math.Log(logitsSoftmax[i])
                );
            }
        }
    }
}