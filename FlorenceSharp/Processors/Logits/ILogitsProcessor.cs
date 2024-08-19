using Microsoft.ML.OnnxRuntime.Tensors;

namespace FlorenceSharp.Processors.Logits
{
    public interface ILogitsProcessor
    {
        public void ProcessLogits(ref DenseTensor<float> logits, DenseTensor<long> inputIDs);
    }
}