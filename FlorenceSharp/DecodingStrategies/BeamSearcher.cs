using System;
using System.Buffers;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using FlorenceSharp.Caching;
using FlorenceSharp.Configs;
using FlorenceSharp.StoppingCriteria;
using Microsoft.ML.OnnxRuntime;
using ONNX.Common.Helpers;
using ONNX.Common.Tensor;

namespace FlorenceSharp.DecodingStrategies
{
    internal struct BeamSearcher<ConfigT> where ConfigT : struct, IFlorenceConfiguration
    {
        // https://huggingface.co/spaces/m-ric/beam_search_visualizer

        // Visualization Settings

        // Number of steps: It is supposed to be 1024, but max is 12 for the visualizer
        // Number of beams: 3
        // Length penalty: 1.0
        // Number of return sequences: 1

        // Values are derived from IDefaultFlorence2Config ( Refer to comment link in it )

        public readonly struct SampleResult(
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
                // Negate for descending order
                return -CumulativeTokenScore.CompareTo(other.CumulativeTokenScore);
            }
        }

        public struct Beam
        {
            private readonly long[] GeneratedTokenIDs;
            
            public double CumulativeScore { get; private set; }

            private readonly DecoderPastKeyValuesCollection<ConfigT> DecoderPastKeyValuesCollection;

            public Beam(long eosTokenID, long bosTokenID)
            {
                var generatedTokenIDs = GeneratedTokenIDs = GC.AllocateUninitializedArray<long>(
                    length: (int) ConfigT.MaxLength,
                    pinned: true);
                
                // Mainly for debugging purposes
                generatedTokenIDs.AsSpan().Fill(-1);
                
                generatedTokenIDs[0]= eosTokenID;
                generatedTokenIDs[1] = bosTokenID;
                
                // Score is accumulated via addition
                // https://chatgpt.com/share/6ab42166-0ae3-4c41-99b0-f9aec6c0a1d6
                CumulativeScore = 0;
                
                DecoderPastKeyValuesCollection = new();
            }
            
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public Memory<long> GetCurrentStepSlice(int currentStepIndex)
            {
                Debug.Assert(currentStepIndex >= INITIAL_STEP_INDEX);
                
                return GeneratedTokenIDs.AsMemory(0, currentStepIndex);
            }

            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public Memory<long> GetLatestTokenSlice(int currentStepIndex)
            {
                Debug.Assert(currentStepIndex >= INITIAL_STEP_INDEX);
                
                return GeneratedTokenIDs.AsMemory(currentStepIndex - 1, 1);
            }

            public void PopulateInputAndOutputWithDecoderPastKeyValuesAndSwapBuffers(
                List<NamedOnnxValue> inputs,
                List<NamedOnnxValue> outputs,
                int currentStepIndex)
            {
                DecoderPastKeyValuesCollection.PopulateInputAndOutputWithPastKeyValuesAndSwapBuffers(
                    inputs,
                    outputs,
                    currentStepIndex
                );
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
                ref var sourceBeam = ref beams[result.ParentBeamIndex];
                
                var sourceTokenIDs = sourceBeam.GeneratedTokenIDs;
                
                var destinationTokenIDs = GeneratedTokenIDs;
                
                sourceTokenIDs.AsSpan(0, currentStepIndex).CopyTo(destinationTokenIDs);
                
                destinationTokenIDs[currentStepIndex] = result.GeneratedTokenID;
                
                CumulativeScore = result.CumulativeTokenScore;
                
                sourceBeam.DecoderPastKeyValuesCollection.CopyTo(in DecoderPastKeyValuesCollection, currentStepIndex);
            }

            public bool BackingMemoryEquals(SampleResult result, Beam[] beams)
            {
                return Unsafe.AreSame(ref this, ref beams[result.ParentBeamIndex]);
            }

            public bool IsDone(
                in FlorenceStopCriteria<ConfigT> stoppingCriteria,
                int currentStepIndex)
            {
                return stoppingCriteria.IsDone(
                    inputIDs: GeneratedTokenIDs.AsSpan(0, currentStepIndex)
                );
            }
            
            public void ResetCumulativeScore()
            {
                CumulativeScore = 0;
            }
        }

        private struct Hypothesis
        {
            private readonly long[] GeneratedTokenIDs;

            private int Length;
            
            // FinalScore takes into account of length penalty
            public double FinalScore { get; private set; }

            public Hypothesis()
            {
                GeneratedTokenIDs = GC.AllocateUninitializedArray<long>(
                    length: (int) ConfigT.MaxLength,
                    pinned: true);
                
                Length = -1;
                
                FinalScore = double.MinValue;
            }

            public void Apply(Beam beam, int currentStepIndex)
            {
                var currentSlice = beam.GetCurrentStepSlice(currentStepIndex);
                
                currentSlice.CopyTo(GeneratedTokenIDs);
                
                Length = currentStepIndex;
                
                // score = sum_logprobs / (hyp.shape[-1] ** self.length_penalty)
                // https://huggingface.co/transformers/v3.5.1/_modules/transformers/generation_beam_search.html
                FinalScore = beam.CumulativeScore / Math.Pow(Length, ConfigT.LengthPenalty);
            }

            public Memory<long> GetMemorySlice()
            {
                return GeneratedTokenIDs.AsMemory(0, Length);
            }
        }

        private struct HypothesisCollection
        {
            private readonly Hypothesis[] Hypotheses;
            
            private int Count;
            
            public HypothesisCollection()
            {
                var hypotheses = Hypotheses = GC.AllocateUninitializedArray<Hypothesis>(
                    length: (int) ConfigT.NumBeams,
                    pinned: true);
                
                for (int i = 0; i < hypotheses.Length; i++)
                {
                    hypotheses[i] = new();
                }
                
                Count = 0;
            }
            
            public bool IsDone => Count >= ConfigT.NumBeams;

            public void Add(Beam beam, int currentStepIndex)
            {
                var writeIndex = Count++;
                
                Debug.Assert(Count <= ConfigT.NumBeams);
                
                ref var currentHypothesis = ref Hypotheses[writeIndex];
                
                currentHypothesis.Apply(beam, currentStepIndex);
            }
            
            public Hypothesis GetBestHypothesis()
            {
                Debug.Assert(IsDone);
                
                var hypotheses = Hypotheses;
                
                var bestHypothesis = hypotheses[0];
                
                for (int i = 1; i < Count; i++)
                {
                    var currentHypothesis = hypotheses[i];
                    
                    if (currentHypothesis.FinalScore > bestHypothesis.FinalScore)
                    {
                        bestHypothesis = currentHypothesis;
                    }
                }
                
                return bestHypothesis;
            }
            
            public void Clear()
            {
                Count = 0;
            }
        }
        
        public uint NumBeams => ConfigT.NumBeams;
        
        private readonly SampleResult[] SampledResults;

        private readonly Beam[] Beams;
        
        private readonly HashSet<int> OutstandingBeamIndices;

        private readonly List<SampleResult> SampleResultsWithDuplicateBeamIndex;

        private readonly EncoderPastKeyValuesCollection<ConfigT> EncoderPastKeyValues;
        
        private HypothesisCollection Hypotheses;

        public BeamSearcher()
        {
            throw new NotSupportedException();
        }
        
        public BeamSearcher(long eosTokenID, long bosTokenID)
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
            
            for (int i = 0; i < beams.Length; i++)
            {
                beams[i] = new(eosTokenID, bosTokenID);
            }
            
            EncoderPastKeyValues = new();

            Hypotheses = new();
        }

        private static void Sample(
            ref Beam beam,
            int currentBeamIndex,
            int currentStepIndex,
            SampleResult[] sampledResults,
            // Required inputs for decoder
            ManagedTensor<long> encoderAttentionMask,
            ManagedTensor<float> encoderHiddenStates,
            in EncoderPastKeyValuesCollection<ConfigT> encoderPastKeyValues,
            int encoderSequenceLength,
            bool useCacheBranch,
            Florence2<ConfigT> florence2)
        {
            var inputIDs = beam.GetCurrentStepSlice(currentStepIndex);

            Memory<long> embeddedInputIDs;
            
            // It is better to pass in a constant useCacheBranch param,
            // so that branches based on it become free when this method is inlined.
            
            Debug.Assert(useCacheBranch == (currentStepIndex > INITIAL_STEP_INDEX));

            if (useCacheBranch)
            {
                embeddedInputIDs = beam.GetLatestTokenSlice(currentStepIndex);
            }

            else
            {
                Debug.Assert(inputIDs.Length == 2);
                embeddedInputIDs = inputIDs;
                beam.ResetCumulativeScore();
            }
            
            var inputEmbeds = florence2.GenerateEmbeddingsForInputIDs(embeddedInputIDs, batchSize: 1);
            
            // Decode into logits
            
            // * 2 cuz key + value
            var listSize = unchecked((int) ((ConfigT.EncoderLayers + ConfigT.DecoderLayers) * 2));
                
            // TODO: Cache this list so we don't have to allocate it every time.
            var pastKeyValues = new List<NamedOnnxValue>(listSize);
            var presentKeyValues = new List<NamedOnnxValue>(listSize);

            beam.PopulateInputAndOutputWithDecoderPastKeyValuesAndSwapBuffers(
                pastKeyValues,
                presentKeyValues,
                currentStepIndex);
                
            encoderPastKeyValues.PopulateInputPastKeyValues(
                pastKeyValues,
                presentKeyValues,
                encoderSequenceLength,
                useCacheBranch
            );

            Debug.Assert(pastKeyValues.Count == listSize);

            var logits = florence2.DecodeIntoLogits(
                encoderAttentionMask,
                encoderHiddenStates,
                inputEmbeds,
                pastKeyValues,
                presentKeyValues,
                useCacheBranch
            );

            if (!useCacheBranch)
            {
                // For the initial step where we do NOT use cache, we input first two tokens
                // </s> and <s> into the model. This means that the logits tensor will have a dimension of
                // [ 1 ( batch_size ), 2 ( Number of input IDs ), vocab_size ].
                
                // Suppose currentStepIndex is 1, we want the logits for the token at index 0
                // ( Tensor indexers are still 0-based, so we need to subtract 1 from currentStepIndex )
                // (currentStepIndex - 1)..currentStepIndex represents a range from
                // (currentStepIndex - 1) to currentStepIndex  ( Exclusive ), which means effectively just ( currentStepIndex - 1 ).
                
                var rangeAll = NRange.All;

                var stepRange = new NRange((currentStepIndex - 1)..currentStepIndex);

                #if DEBUG
                Debug.Assert((stepRange.End.Value - stepRange.Start.Value) == 1);
                
                var oldLogits = logits;
                #endif
                
                logits = logits.SNTensor.Slice([rangeAll, stepRange, rangeAll]);
                
                #if DEBUG
                const int LOGIT_TENSOR_SIZE = 51289;
                
                var accurateSlicedTensor = new ManagedTensor<float>(
                    (ReadOnlySpan<nint>) [ 1, 1, LOGIT_TENSOR_SIZE ],
                    initialize: false
                );
                
                for (int i = 0; i < LOGIT_TENSOR_SIZE; i++)
                {
                    accurateSlicedTensor.OnnxDenseTensor[0, 0, i] = oldLogits.OnnxDenseTensor[0, 1, i];
                }
                
                Debug.Assert(accurateSlicedTensor.ValuesArr.SequenceEqual(logits.ValuesArr));
                #endif
            }
            
            // Process the logits
            
            ref readonly var logitsProcessor = ref florence2.LogitsProcessor;
            
            logitsProcessor.ProcessLogits(logits.ValuesArr.AsMemory(), inputIDs);
                        
            // Get the TopK results
            
            var topKResult = logits.TopK(ConfigT.TopK);

            Debug.Assert(topKResult.Logits.ValuesArr.Length == ConfigT.TopK);
            
            var logitsSoftmax = topKResult.Logits.SoftMaxInPlace().ValuesArr;
            
            var topKIndices = topKResult.Indices.ValuesArr;

            var numBeams = ConfigT.NumBeams;
            
            for (int i = 0; i < numBeams; i++)
            {
                var tokenID = topKIndices[i];

                var logProb = Math.Log(logitsSoftmax[i]);
                
                sampledResults[(currentBeamIndex * numBeams) + i] = new(
                    generatedTokenId: (int) tokenID,
                    cumulativeTokenScore: beam.CumulativeScore + logProb,
                    parentBeamIndex: currentBeamIndex
                );
            }
        }
        
        private const int INITIAL_STEP_INDEX = 2;
        
        public Memory<long> Search(ManagedTensor<float> 
                encoderHiddenStates,
                ManagedTensor<long> encoderAttentionMask,
                Florence2<ConfigT> florence2,
                in FlorenceStopCriteria<ConfigT> stoppingCriteria,
                int encoderSequenceLength)
        {
            var beams = Beams;
            
            var numBeams = beams.Length;
            
            // Initialize initial beams
            
            // For the first sequence, we will have only one beam

            ref var initialBeam = ref beams[0];
            
            #if DEBUG
            var eosTokenID = florence2.EndOfSequenceTokenID;
            var bosTokenID = florence2.BeginningOfSequenceTokenID;
            
            initialBeam.GetCurrentStepSlice(INITIAL_STEP_INDEX).Span.SequenceEqual([ eosTokenID, bosTokenID ]);
            #endif
            
            // Sample the first beam
            
            // currentBeamIndex is 0-based, while currentStepIndex is NOT
            
            var sampledResults = SampledResults;

            ref readonly var encoderPastKeyValues = ref EncoderPastKeyValues;

            var currentStepIndex = INITIAL_STEP_INDEX;
            
            Sample(
                ref initialBeam,
                currentBeamIndex: 0,
                currentStepIndex,
                sampledResults,
                encoderAttentionMask,
                encoderHiddenStates,
                in encoderPastKeyValues,
                encoderSequenceLength,
                useCacheBranch: false,
                florence2);
            
            // Create the 3 beams

            int currentBeamIndex;

            for (currentBeamIndex = 0; currentBeamIndex < numBeams; currentBeamIndex++)
            {
                ref var currentBeam = ref beams[currentBeamIndex];
                
                // currentBeam.AppendSampleResult(sampledResults[currentBeamIndex], currentStepIndex: INITIAL_STEP_INDEX);
                
                currentBeam.OverwriteWithSampleResult(
                    sampledResults[currentBeamIndex],
                    currentStepIndex,
                    beams);
            }
            
            var outstandingBeamIndices = OutstandingBeamIndices;
            
            var sampleResultsWithDuplicateBeamIndex = SampleResultsWithDuplicateBeamIndex;
            
            ref var hypothesisCollection = ref Hypotheses;
            
            // Clear Hypotheses
            hypothesisCollection.Clear();
            
            for (++currentStepIndex; 
                 true; // This helps me understand better, even though it can be optimized away
                 currentStepIndex++)
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
                        ref currentBeam,
                        currentBeamIndex,
                        currentStepIndex,
                        sampledResults,
                        encoderAttentionMask,
                        encoderHiddenStates,
                        in encoderPastKeyValues,
                        encoderSequenceLength,
                        useCacheBranch: true,
                        florence2);
                }
                
                // Sort sampled results based on cumulative score.
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
                        
                        parentBeam.AppendSampleResult(result, currentStepIndex);
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

                var sampleResultWithDuplicateBeamIndex = 0;
                
                foreach (var beamIndex in outstandingBeamIndices)
                {
                    ref var currentBeam = ref beams[beamIndex];

                    var sampleResult = sampleResultsWithDuplicateBeamIndex[sampleResultWithDuplicateBeamIndex++];
                    
                    currentBeam.OverwriteWithSampleResult(
                        sampleResult,
                        currentStepIndex: currentStepIndex,
                        beams);
                }

                // Should a beam be done, we add it to Hypotheses. The beam then takes the next best sample.
                
                // Say we usually take 3 ( 1 for each beam ). [ 4, 3, 2 ]
                // Beams:     1  2  3
                // Samples: [ 4, 3, 2, 1, 0 ]
                
                //                              Done           Take the next highest sample, which is 1.
                //                              v              v
                // Suppose beam 2 is done. [ 4, 3, 2 ] -> [ 4, 1, 2 ]
                
                // So yeah, the new beamIndex starts from numBeams, and increments whenever a beam is done.
                var newBeamIndex = numBeams;
                
                for (currentBeamIndex = 0; currentBeamIndex < numBeams; currentBeamIndex++)
                {
                    ref var currentBeam = ref beams[currentBeamIndex];
                    
                    var isCurrentBeamDone = currentBeam.IsDone(stoppingCriteria, currentStepIndex);
                    
                    if (isCurrentBeamDone)
                    {
                        // This copies data from beam to hypothesis memory, therefore freeing up beam memory
                        hypothesisCollection.Add(currentBeam, currentStepIndex);
                        
                        // We still want to sample numBeams number of samples. Since the current beam is done,
                        // we will take the next best sample.
                        
                        var newSample = sampledResults[newBeamIndex++];

                        // No side-effect. As mentioned above, the beam memory is freed up.
                        // We also guarantee that only a single beam points to a single beam memory!
                        // ( Samples can point to the same beam memory, but not beams )
                        if (currentBeam.BackingMemoryEquals(newSample, beams))
                        {
                            currentBeam.AppendSampleResult(
                                newSample,
                                currentStepIndex: currentStepIndex);
                        }

                        else
                        {
                            currentBeam.OverwriteWithSampleResult(
                                newSample,
                                currentStepIndex: currentStepIndex,
                                beams);
                        }
                        if (hypothesisCollection.IsDone)
                        {
                            return hypothesisCollection
                                .GetBestHypothesis()
                                .GetMemorySlice();
                        }
                    }
                }
            }
        }
    }
}