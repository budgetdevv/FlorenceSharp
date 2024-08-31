using System;
using FlorenceSharp.Configs;

namespace FlorenceSharp.StoppingCriteria
{
    public readonly struct FlorenceStopCriteria<ConfigT>(long endOfSequenceTokenId): IStoppingCriterion
        where ConfigT : IFlorenceGenerationConfiguration
    {
        private readonly MaxLengthStopCriterion<ConfigT> MaxLengthStopCriterion = new();
        
        private readonly EOSTokenStopCriterion EOSTokenStopCriterion = new(endOfSequenceTokenId);

        public bool IsDone(ReadOnlySpan<long> inputIDs)
        {
            return MaxLengthStopCriterion.IsDone(inputIDs) ||
                   EOSTokenStopCriterion.IsDone(inputIDs);
        }
    }
}