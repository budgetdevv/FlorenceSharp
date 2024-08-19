using System;

namespace FlorenceSharp.StoppingCriteria
{
    public readonly struct EOSTokenStopCriterion: IStoppingCriterion
    {
        public bool IsDone(ReadOnlySpan<long> inputIDs, ReadOnlySpan<double> scores)
        {
            throw new NotImplementedException();
        }
    }
}