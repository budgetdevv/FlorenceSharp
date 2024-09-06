using System;

namespace FlorenceSharp.Processors.Logits
{
    public interface ILogitsProcessor
    {
        public void ProcessLogits(Memory<float> logits, Memory<long> inputIDs);
    }
}