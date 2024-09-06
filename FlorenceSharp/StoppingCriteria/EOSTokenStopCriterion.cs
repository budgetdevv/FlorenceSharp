using System;
using System.Runtime.CompilerServices;

namespace FlorenceSharp.StoppingCriteria
{
    public readonly struct EOSTokenStopCriterion: IStoppingCriterion
    {
        private readonly long EOSTokenID;
     
        public EOSTokenStopCriterion()
        {
            throw new NotImplementedException();
        }
        
        public EOSTokenStopCriterion(long eosTokenID)
        {
            EOSTokenID = eosTokenID;
        }
        
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public bool IsDone(ReadOnlySpan<long> inputIDs)
        {
            return inputIDs[^1] == EOSTokenID;
        }
    }
}