using System;

namespace FlorenceSharp.StoppingCriteria
{
    public readonly struct FlorenceStopCriteria: IStoppingCriterion
    {
        private readonly MaxLengthStopCriterion MaxLengthStopCriterion;
        
        private readonly EOSTokenStopCriterion EOSTokenStopCriterion;

        public bool IsDone(ReadOnlySpan<long> inputIDs, ReadOnlySpan<double> scores)
        {
            return MaxLengthStopCriterion.IsDone(inputIDs, scores) ||
                   EOSTokenStopCriterion.IsDone(inputIDs, scores);
        }
    }
}