using System;
using FlorenceSharp.Configs;

namespace FlorenceSharp.StoppingCriteria
{
    public readonly struct FlorenceStopCriteria<ConfigT>: IStoppingCriterion
        where ConfigT : IFlorenceGenerationConfiguration
    {
        private readonly MaxLengthStopCriterion<ConfigT> MaxLengthStopCriterion;

        private readonly EOSTokenStopCriterion EOSTokenStopCriterion;

        public FlorenceStopCriteria()
        {
            throw new NotImplementedException();
        }

        public FlorenceStopCriteria(long endOfSequenceTokenId)
        {
            MaxLengthStopCriterion = new();
            EOSTokenStopCriterion = new(endOfSequenceTokenId);
        }
        
        public bool IsDone(ReadOnlySpan<long> inputIDs)
        {
            return MaxLengthStopCriterion.IsDone(inputIDs) ||
                   EOSTokenStopCriterion.IsDone(inputIDs);
        }
    }
}