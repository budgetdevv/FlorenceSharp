using Microsoft.ML.OnnxRuntime.Tensors;

namespace FlorenceSharp.Processors.Logits
{
    public readonly struct FlorenceLogitsProcessor: ILogitsProcessor
    {
        // https://imgur.com/a/6C0VkXA
        // https://huggingface.co/transformers/v3.5.1/internal/generation_utils.html
        // https://huggingface.co/transformers/v4.1.1/_modules/transformers/generation_logits_process.html
        
        private readonly NoRepeatNGramLogitsProcessor NoRepeatNGramLogitsProcessor;
        
        private readonly ForcedBOSTokenLogitsProcessor ForcedBOSTokenLogitsProcessor;
        
        private readonly ForcedEOSTokenLogitsProcessor ForcedEOSTokenLogitsProcessor;

        public FlorenceLogitsProcessor()
        {
            NoRepeatNGramLogitsProcessor = new();
            ForcedBOSTokenLogitsProcessor = new();
            ForcedEOSTokenLogitsProcessor = new();
        }
        
        public void ProcessLogits(ref DenseTensor<float> logits, DenseTensor<long> inputIDs)
        {
            NoRepeatNGramLogitsProcessor.ProcessLogits(ref logits, inputIDs);
            ForcedBOSTokenLogitsProcessor.ProcessLogits(ref logits, inputIDs);
            ForcedEOSTokenLogitsProcessor.ProcessLogits(ref logits, inputIDs);
        }
    }
}