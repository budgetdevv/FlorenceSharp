using System;
using FlorenceSharp.Configs;
using FlorenceSharp.Tokenizers;

namespace FlorenceSharp.Processors.Logits
{
    public readonly struct FlorenceLogitsProcessor<ConfigT>: ILogitsProcessor
        where ConfigT: IFlorenceGenerationConfiguration
    {
        // https://imgur.com/a/6C0VkXA
        // https://huggingface.co/transformers/v3.5.1/internal/generation_utils.html
        // https://huggingface.co/transformers/v4.1.1/_modules/transformers/generation_logits_process.html
        
        private readonly NoRepeatNGramLogitsProcessor<ConfigT> NoRepeatNGramLogitsProcessor;
        
        private readonly ForcedBOSTokenLogitsProcessor ForcedBOSTokenLogitsProcessor;
        
        private readonly ForcedEOSTokenLogitsProcessor ForcedEOSTokenLogitsProcessor;

        public FlorenceLogitsProcessor()
        {
            throw new NotImplementedException();
        }
        
        public FlorenceLogitsProcessor(in FlorenceBartTokenizer tokenizer)
        {
            NoRepeatNGramLogitsProcessor = new();
            ForcedBOSTokenLogitsProcessor = new(in tokenizer);
            ForcedEOSTokenLogitsProcessor = new();
        }
        
        public void ProcessLogits(Memory<float> logits, Memory<long> inputIDs)
        {
            NoRepeatNGramLogitsProcessor.ProcessLogits(logits, inputIDs);
            ForcedBOSTokenLogitsProcessor.ProcessLogits(logits, inputIDs);
            ForcedEOSTokenLogitsProcessor.ProcessLogits(logits, inputIDs);
        }
    }
}