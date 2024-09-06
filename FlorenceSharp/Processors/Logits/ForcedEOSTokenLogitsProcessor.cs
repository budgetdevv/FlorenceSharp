using System;

namespace FlorenceSharp.Processors.Logits
{
    public readonly struct ForcedEOSTokenLogitsProcessor: ILogitsProcessor
    {
        public void ProcessLogits(Memory<float> logits, Memory<long> inputIDs)
        {
        }
    }
}