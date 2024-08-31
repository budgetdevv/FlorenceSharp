using System;
using System.Runtime.CompilerServices;
using FlorenceSharp.Configs;
using FlorenceSharp.Tokenizers;

namespace FlorenceSharp.StoppingCriteria
{
    public readonly struct EOSTokenStopCriterion(long endOfSentenceTokenId): IStoppingCriterion
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public bool IsDone(ReadOnlySpan<long> inputIDs)
        {
            return inputIDs[^1] == endOfSentenceTokenId;
        }
    }
}