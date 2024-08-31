using System;
using System.Buffers;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using FlorenceSharp.Collections;
using FlorenceSharp.Configs;
using FlorenceSharp.Helpers;
using FlorenceSharp.StoppingCriteria;
using FlorenceSharp.Tensor;
using FlorenceSharp.Tokenizers;

namespace FlorenceSharp.DecodingStrategies
{
    public readonly struct BeamSearcher<ConfigT> where ConfigT : struct, IFlorenceConfiguration
    {
        // https://huggingface.co/spaces/m-ric/beam_search_visualizer

        // Visualization Settings

        // Number of steps: It is supposed to be 1024, but max is 12 for the visualizer
        // Number of beams: 3
        // Length penalty: 1.0
        // Number of return sequences: 1

        // Values are derived from IDefaultFlorence2Config ( Refer to comment link in it )

        private readonly struct SampleResult(
            int generatedTokenId,
            double cumulativeTokenScore,
            int parentBeamIndex): IComparable<SampleResult>
        {
            public readonly int GeneratedTokenID = generatedTokenId;

            public readonly double CumulativeTokenScore = cumulativeTokenScore;
            
            public readonly int ParentBeamIndex = parentBeamIndex;
            
            // Used for sorting
            public int CompareTo(SampleResult other)
            {
                return CumulativeTokenScore.CompareTo(other.CumulativeTokenScore);
            }
        }

        private struct Beam
        {
            private readonly long[] GeneratedTokenIDs;
            
            public double CumulativeScore { get; private set; }

            public bool IsDone { get; private set; }

            public Beam(int endOfSequenceID)
            {
                var generatedTokenIDs = GeneratedTokenIDs = GC.AllocateUninitializedArray<long>(
                    length: (int) ConfigT.MaxLength,
                    pinned: true);
                
                // Write as first element
                MemoryMarshal.GetArrayDataReference(generatedTokenIDs) = endOfSequenceID;
                
                CumulativeScore = 0;
                
                IsDone = false;
            }

            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public Memory<long> GetCurrentStepSlice(int currentStepIndex)
            {
                return GeneratedTokenIDs.AsMemory(0, currentStepIndex);
            }

            public void AppendSampleResult(SampleResult result, int currentStepIndex)
            {
                var generatedTokenIDs = GeneratedTokenIDs;
                
                generatedTokenIDs[currentStepIndex] = result.GeneratedTokenID;
                
                CumulativeScore = result.CumulativeTokenScore;
            }
            
            public void OverwriteWithSampleResult(
                SampleResult result,
                int currentStepIndex,
                Beam[] beams)
            {
                var sourceBeam = beams[result.ParentBeamIndex];
                
                var sourceTokenIDs = sourceBeam.GeneratedTokenIDs;
                
                var destinationTokenIDs = GeneratedTokenIDs;
                
                sourceTokenIDs.AsSpan(0, currentStepIndex).CopyTo(destinationTokenIDs);
                
                destinationTokenIDs[currentStepIndex] = result.GeneratedTokenID;
                
                CumulativeScore = result.CumulativeTokenScore;
            }

            public bool UpdateAndReturnDoneState(
                in FlorenceStopCriteria<ConfigT> stoppingCriteria,
                int currentStepIndex)
            {
                return IsDone = stoppingCriteria.IsDone(
                    inputIDs: GeneratedTokenIDs.AsSpan(0, currentStepIndex)
                );
            }
        }
        
        public uint NumBeams => ConfigT.NumBeams;
        
        private readonly SampleResult[] SampledResults;

        private readonly Beam[] Beams;
        
        private readonly HashSet<int> OutstandingBeamIndices;

        private readonly List<SampleResult> SampleResultsWithDuplicateBeamIndex;
        
        private const string EOS_TOKEN = FlorenceSpecialTokens.END_OF_SEQUENCE;

        private readonly long EndOfSequenceTokenID;
        
        public BeamSearcher(in FlorenceBartTokenizer tokenizer)
        {
            var numBeams = (int) NumBeams;
            
            // Long-term allocation, allocate in POH
            
            // https://imgur.com/a/yTXkPzZ
            // As shown, at any point in time there's a maximum of numBeams * numBeams outstanding beams.
            // We will pick numBeams best beams ( Highlighted in yellow ) from these outstanding beams.
            // OutstandingBeams = new Beam[numBeams * numBeams];
            SampledResults = GC.AllocateUninitializedArray<SampleResult>(
                length: numBeams * numBeams,
                pinned: true);
            
            var beams = Beams = GC.AllocateUninitializedArray<Beam>(
                length: numBeams,
                pinned: true);

            OutstandingBeamIndices = new(numBeams);
            
            SampleResultsWithDuplicateBeamIndex = new(numBeams);
            
            EndOfSequenceTokenID = tokenizer.GetTokenID(EOS_TOKEN);
            
            // Initialize beams. We only have to do this once.
            var endOfSequenceTokenID = (int) EndOfSequenceTokenID;
            
            for (int i = 0; i < beams.Length; i++)
            {
                beams[i] = new(endOfSequenceTokenID);
            }
        }

        private static void Sample(
            Beam beam, 
            int currentBeamIndex,
            int currentStepIndex,
            SampleResult[] sampledResults,
            // Required inputs for decoder
            ManagedTensor<long> encoderAttentionMask,
            ManagedTensor<float> encoderHiddenStates,
            in Florence2<ConfigT> florence2)
        {
            // Get embeddings for the generated tokens

            var inputIDs = beam.GetCurrentStepSlice(currentStepIndex);

            // TODO: Embed with multiple batches
            var inputEmbeds = florence2.GenerateEmbeddingsForInputIDs(inputIDs, batchSize: 1);
            
            // Decode into logits

            var logits = florence2.DecodeIntoLogits(
                encoderAttentionMask,
                encoderHiddenStates,
                inputEmbeds);
            
            // Get the TopK results
            
            var topKResult = logits.TopK(ConfigT.TopK);

            var logitsSoftmax = topKResult.Logits.SoftMaxInPlace().ValuesArr;
            
            var topKIndices = topKResult.Indices.ValuesArr;

            var numBeams = ConfigT.NumBeams;
            
            for (int i = 0; i < numBeams; i++)
            {
                var tokenID = topKIndices[i];
                
                sampledResults[currentBeamIndex + i] = new(
                    generatedTokenId: (int) tokenID,
                    cumulativeTokenScore: Math.Log(logitsSoftmax[i]),
                    parentBeamIndex: currentBeamIndex
                );
            }
        }
        
        public Memory<long> Search(
            ManagedTensor<long> encoderAttentionMask,
            ManagedTensor<float> encoderHiddenStates,
            in Florence2<ConfigT> florence2,
            in FlorenceBartTokenizer tokenizer,
            in FlorenceStopCriteria<ConfigT> stoppingCriteria)
            // out int outputTokens)
        {
            var eosTokenID = EndOfSequenceTokenID;

            var beams = Beams;
            
            var numBeams = beams.Length;
            
            // Initialize initial beams
            
            // For the first sequence, we will have only one beam

            ref var initialBeam = ref beams[0];
            
            // Sample the first beam
            
            // currentBeamIndex is 0-based, while currentStepIndex is NOT
            
            var sampledResults = SampledResults;

            const int INITIAL_STEP_INDEX = 1;
            
            Sample(
                initialBeam,
                currentBeamIndex: 0,
                currentStepIndex: INITIAL_STEP_INDEX,
                sampledResults,
                encoderAttentionMask,
                encoderHiddenStates,
                florence2);
            
            // Create the 3 beams

            int currentBeamIndex;

            for (currentBeamIndex = 0; currentBeamIndex < numBeams; currentBeamIndex++)
            {
                ref var currentBeam = ref beams[currentBeamIndex];
                
                currentBeam.AppendSampleResult(sampledResults[currentBeamIndex], currentStepIndex: INITIAL_STEP_INDEX);
            }
            
            var outstandingBeamIndices = OutstandingBeamIndices;
            
            var sampleResultsWithDuplicateBeamIndex = SampleResultsWithDuplicateBeamIndex;
            
            for (int currentStepIndex = INITIAL_STEP_INDEX + 1; true; currentStepIndex++)
            {
                // Now the fun begins
                
                // Refresh outstandingBeamIndices
                for (int i = 0; i < numBeams; i++)
                {
                    outstandingBeamIndices.Add(i);
                }
                
                // Clear sampleResultsWithDuplicateBeamIndex
                sampleResultsWithDuplicateBeamIndex.Clear();
                
                // Sample beams
                for (currentBeamIndex = 0; currentBeamIndex < numBeams; currentBeamIndex++)
                {
                    ref var currentBeam = ref beams[currentBeamIndex];
                
                    Sample(
                        currentBeam,
                        currentBeamIndex: currentBeamIndex,
                        currentStepIndex: currentStepIndex,
                        sampledResults,
                        encoderAttentionMask,
                        encoderHiddenStates,
                        florence2);
                }
                
                Array.Sort(sampledResults);
                
                // Pick the best beams ( We only require numBeams amount )
                
                for (int i = 0; i < numBeams; i++)
                {
                    var result = sampledResults[i];
                    
                    var parentBeamIndex = result.ParentBeamIndex;
                    
                    // Is the beam memory available?
                    if (outstandingBeamIndices.Remove(parentBeamIndex))
                    {
                        ref var parentBeam = ref beams[parentBeamIndex];
                        
                        parentBeam.AppendSampleResult(result, currentStepIndex: INITIAL_STEP_INDEX);
                    }
                    
                    // Otherwise, we add it to sampleResultsWithDuplicateBeamIndex
                    else
                    {
                        sampleResultsWithDuplicateBeamIndex.Add(result);
                    }
                }
                
                // Handle duplicates
                
                // Whenever there's a duplicate, we end up NOT removing anything from outstandingBeamIndices.
                // So in theory, sampleResultsWithDuplicateBeamIndex.Count should be equal to outstandingBeamIndices.Count.
                Debug.Assert(sampleResultsWithDuplicateBeamIndex.Count == outstandingBeamIndices.Count);

                var sampleResultIndex = 0;
                
                foreach (var beamIndex in outstandingBeamIndices)
                {
                    ref var currentBeam = ref beams[beamIndex];

                    var sampleResult = sampleResultsWithDuplicateBeamIndex[sampleResultIndex];
                    
                    currentBeam.OverwriteWithSampleResult(
                        sampleResult,
                        currentStepIndex: currentStepIndex,
                        beams);
                }

                var isDone = false;

                for (currentBeamIndex = 0; currentBeamIndex < numBeams; currentBeamIndex++)
                {
                    ref var currentBeam = ref beams[currentBeamIndex];
                    
                    isDone |= currentBeam.UpdateAndReturnDoneState(stoppingCriteria, currentStepIndex);
                }

                if (isDone)
                {
                    // The first beam always contain the best result

                    return initialBeam.GetCurrentStepSlice(currentStepIndex);
                }
            }
        }
    }
}