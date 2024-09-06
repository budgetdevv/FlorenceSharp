using System;
using FlorenceSharp.Tokenizers;

namespace FlorenceSharp.Processors.Logits
{
    public readonly struct ForcedBOSTokenLogitsProcessor: ILogitsProcessor
    {
        private readonly int bosTokenID;
        
        public ForcedBOSTokenLogitsProcessor(in FlorenceBartTokenizer tokenizer)
        {
            bosTokenID = tokenizer.GetTokenID(FlorenceSpecialTokens.BEGINNING_OF_SEQUENCE);
        }
        
        public void ProcessLogits(Memory<float> logits, Memory<long> inputIDs)
        {
            // https://huggingface.co/docs/transformers/v4.15.0/en/internal/generation_utils#transformers.ForcedBOSTokenLogitsProcessor
            // https://github.com/huggingface/transformers/blob/v4.15.0/src/transformers/generation_logits_process.py#L559
            
            if (inputIDs.Length != 1)
            {
                return;
            }
            
            var logitsSpan = logits.Span;
            
            logitsSpan.Fill(float.NegativeInfinity);
            logitsSpan[bosTokenID] = 0;
        }
    }
}