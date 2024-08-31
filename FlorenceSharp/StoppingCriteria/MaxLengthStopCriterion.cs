using System;
using System.Runtime.CompilerServices;
using FlorenceSharp.Configs;

namespace FlorenceSharp.StoppingCriteria
{
    public readonly struct MaxLengthStopCriterion<ConfigT>: IStoppingCriterion
        where ConfigT: IFlorenceGenerationConfiguration
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public bool IsDone(ReadOnlySpan<long> inputIDs)
        {
            var maxLength = ConfigT.MaxLength;
            
            return inputIDs.Length >= maxLength;
        }
    }
}