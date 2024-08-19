using System;

namespace FlorenceSharp.StoppingCriteria
{
    public interface IStoppingCriterion
    {
        public bool IsDone(ReadOnlySpan<long> inputIDs, ReadOnlySpan<double> scores);
    }
}