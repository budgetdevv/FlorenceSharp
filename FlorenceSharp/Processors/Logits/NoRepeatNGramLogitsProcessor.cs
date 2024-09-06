using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using FlorenceSharp.Configs;

namespace FlorenceSharp.Processors.Logits
{
    public readonly struct NoRepeatNGramLogitsProcessor<ConfigT>: ILogitsProcessor
        where ConfigT: IFlorenceGenerationConfiguration
    {
        // We can probably optimize this data-structure better via caching computed ngrams on a per-beam basis.
        
        private readonly struct NGramPrefix: IEquatable<NGramPrefix>
        {
            public readonly Memory<long> PrefixingTokens;
        
            public NGramPrefix(Memory<long> prefixingTokens)
            {
                PrefixingTokens = prefixingTokens;
            }
        
            public bool Equals(NGramPrefix other)
            {
                return PrefixingTokens.Span.SequenceEqual(other.PrefixingTokens.Span);
            }
        
            public override bool Equals(object? obj)
            {
                return obj is NGramPrefix other && Equals(other);
            }
        
            public static bool operator ==(NGramPrefix left, NGramPrefix right)
            {
                return left.Equals(right);
            }
        
            public static bool operator !=(NGramPrefix left, NGramPrefix right)
            {
                return !left.Equals(right);
            }
            
            public override int GetHashCode()
            {
                var accumulator = 0;
        
                foreach (var tokenID in PrefixingTokens.Span)
                {
                    accumulator += unchecked((int) tokenID);
                }
                
                return accumulator;
            }
        }

        private readonly struct BannedTokens
        {
            public readonly List<long> Tokens;

            public BannedTokens(long bannedToken)
            {
                Tokens = [ bannedToken ];
            }
            
            public void Add(long token)
            {
                Tokens.Add(token);
            }
        }
        
        private readonly Dictionary<NGramPrefix, BannedTokens> NGrams;

        public NoRepeatNGramLogitsProcessor()
        {
            NGrams = new();
        }
        
        public void ProcessLogits(Memory<float> logits, Memory<long> inputIDs)
        {
            // https://huggingface.co/docs/transformers/v4.15.0/en/internal/generation_utils#transformers.NoRepeatNGramLogitsProcessor
            // https://github.com/huggingface/transformers/blob/v4.15.0/src/transformers/generation_logits_process.py#L279
            // https://github.com/facebookresearch/fairseq/blob/a07cb6f40480928c9e0548b737aadd36ee66ac76/fairseq/sequence_generator.py#L345
            
            // Generate n-grams from inputIDs
            
            var noRepeatNGramSize = unchecked((int) ConfigT.NoRepeatNGramSize);
            
            var noRepeatNGramPrefixSize = noRepeatNGramSize - 1;;
            
            var inputIDsLength = inputIDs.Length;
            
            if (inputIDsLength >= noRepeatNGramSize)
            {
                var nGrams = NGrams;
            
                // [ 1, 2, 3 ] -> [ 1, 2 , 3 ]
                // Number of iterations: Length ( 3 ) + 1 - NoRepeatNgramSize ( 3 ) = 1
            
                // [ 1, 2, 3, 4 ] -> [ 1, 2 , 3 ], [ 2, 3, 4 ]
                // Number of iterations: Length ( 4 ) + 1 - NoRepeatNgramSize ( 3 ) = 2
            
                for (int i = 0; i < inputIDsLength + 1 - noRepeatNGramSize; i++)
                {
                    var currentSlice = inputIDs.Slice(i, noRepeatNGramSize);
                
                    var prefixingTokens = currentSlice.Slice(0, noRepeatNGramPrefixSize);
                
                    var bannedToken = currentSlice.Span[noRepeatNGramPrefixSize];
                
                    var nGramPrefix = new NGramPrefix(prefixingTokens);

                    ref var currentBannedTokens = ref CollectionsMarshal.GetValueRefOrAddDefault(
                        dictionary: nGrams, 
                        key: nGramPrefix,
                        out var exists);

                    if (exists)
                    {
                        currentBannedTokens.Add(bannedToken);
                    }

                    else
                    {
                        currentBannedTokens = new(bannedToken);
                    }
                }
            
                // Look at the most recent prefixing tokens
                // TODO: Honestly this makes me wonder...do we even require a dictionary?
                // Why not just add to ban list if the current prefixing tokens of the sequence
                // are the same as mostRecentPrefixingTokens?
            
                var mostRecentPrefixingTokens = inputIDs
                    .Slice(inputIDsLength - noRepeatNGramPrefixSize, noRepeatNGramPrefixSize);
            
                var mostRecentNGramPrefix = new NGramPrefix(mostRecentPrefixingTokens);

                if (nGrams.TryGetValue(mostRecentNGramPrefix, out var bannedTokens))
                {
                    var logitsSpan = logits.Span;
            
                    foreach (var bannedToken in bannedTokens.Tokens)
                    {
                        logitsSpan[(int) bannedToken] = float.NegativeInfinity;
                    }
                }
                
                nGrams.Clear();
            }
        }
    }
}