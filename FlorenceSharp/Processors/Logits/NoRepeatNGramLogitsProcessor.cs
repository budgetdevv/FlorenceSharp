using System;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace FlorenceSharp.Processors.Logits
{
    public readonly struct NoRepeatNGramLogitsProcessor: ILogitsProcessor
    {
        public void ProcessLogits(ref DenseTensor<float> logits, DenseTensor<long> inputIDs)
        {
            throw new NotImplementedException();
        }
    }
}