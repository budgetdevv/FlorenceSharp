using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using FlorenceSharp.Configs;
using FlorenceSharp.Helpers;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace FlorenceSharp.Caching
{
    internal static class PastKeyValueNames<ConfigT> where ConfigT: IFlorenceGenerationConfiguration
    {
        public static readonly PastKeyValueNames[] NAMES;
        
        static PastKeyValueNames()
        {
            // We are assuming that they're equal, which means we can store encoder and decoder names under the same array.
            Debug.Assert(ConfigT.EncoderLayers == ConfigT.DecoderLayers);
            
            var pastKeyValuesNames = NAMES = AllocationHelpers
                .AllocatePinnedArrayUninitialized<PastKeyValueNames>(ConfigT.EncoderLayers);

            for (var i = 0; i < pastKeyValuesNames.Length; i++)
            {
                pastKeyValuesNames[i] = new(i);
            }
        }
    }    
    
    internal readonly struct PastKeyValueNames
    {
        // https://imgur.com/a/florence2-decoder-merged-tI0RFxq
        
        private const string 
            PAST_KEY_VALUES_NAME_PREFIX = "past_key_values",
            PRESENT_KEY_VALUES_NAME_PREFIX = "present";
        
        public static readonly PastKeyValueNames[] NAMES;
        
        public readonly string 
            // Inputs
            PastEncoderKeyName,
            PastEncoderValueName,
            PastDecoderKeyName,
            PastDecoderValueName,
            // Outputs
            PresentEncoderKeyName,
            PresentEncoderValueName,
            PresentDecoderKeyName,
            PresentDecoderValueName;
        
        public PastKeyValueNames()
        {
            throw new NotSupportedException();
        }

        public PastKeyValueNames(int currentIndex)
        {
            PastEncoderKeyName = $"{PAST_KEY_VALUES_NAME_PREFIX}.{currentIndex}.encoder.key";
            PastEncoderValueName = $"{PAST_KEY_VALUES_NAME_PREFIX}.{currentIndex}.encoder.value";
            PastDecoderKeyName = $"{PAST_KEY_VALUES_NAME_PREFIX}.{currentIndex}.decoder.key";
            PastDecoderValueName = $"{PAST_KEY_VALUES_NAME_PREFIX}.{currentIndex}.decoder.value";
                
            PresentEncoderKeyName = $"{PRESENT_KEY_VALUES_NAME_PREFIX}.{currentIndex}.encoder.key";
            PresentEncoderValueName = $"{PRESENT_KEY_VALUES_NAME_PREFIX}.{currentIndex}.encoder.value";
            PresentDecoderKeyName = $"{PRESENT_KEY_VALUES_NAME_PREFIX}.{currentIndex}.decoder.key";
            PresentDecoderValueName = $"{PRESENT_KEY_VALUES_NAME_PREFIX}.{currentIndex}.decoder.value";
        }
    }

    internal static class PastKeyValuesHelpers<ConfigT> where ConfigT: IFlorenceGenerationConfiguration
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static int ComputeTensorSizeFast(int sequenceLength)
        {
            // JIT should fold 1, ConfigT.EncoderLayers and 64 into a constant,
            // which means a single multiplication.
            
            // [ batch_size, 16, encoder_sequence_length, 64 ]
            // [ batch_size, 16, decoder_sequence_length, 64 ]
            // https://imgur.com/a/florence2-decoder-merged-tI0RFxq
            
            return unchecked((int) (1 * ConfigT.EncoderAttentionHeads * sequenceLength * 64));
        }
    }
    
    public readonly struct DecoderPastKeyValuesCollection<ConfigT> where ConfigT: IFlorenceGenerationConfiguration
    {
        private struct DecoderKeyValuePair
        {
            public float[]
                // These are for inputting
                PastKey,
                PastValue,
                // These are for outputting
                PresentKey,
                PresentValue;

            public DecoderKeyValuePair()
            {
                var size = PastKeyValuesHelpers<ConfigT>.ComputeTensorSizeFast(unchecked((int) ConfigT.MaxLength));
                
                // Negative infinity is just for debugging purposes -
                // It indicates that a given slot have yet to be written to.
                var fillValue = float.NegativeInfinity;
                
                PastKey = AllocationHelpers.AllocatePinnedArrayAndFill(size, fillValue);
                PastValue = AllocationHelpers.AllocatePinnedArrayAndFill(size, fillValue);
                
                // PresentKeyValuesEncoderKey = AllocationHelpers.AllocatePinnedArrayAndFill(size, fillValue);
                // PresentKeyValuesEncoderValue = AllocationHelpers.AllocatePinnedArrayAndFill(size, fillValue);
                
                PresentKey = AllocationHelpers.AllocatePinnedArrayAndFill(size, fillValue);
                PresentValue = AllocationHelpers.AllocatePinnedArrayAndFill(size, fillValue);
            }
        }

        private readonly DecoderKeyValuePair[] PastKeyValues;

        public DecoderPastKeyValuesCollection()
        {
            var pastKeyValues = PastKeyValues = AllocationHelpers
                .AllocatePinnedArrayUninitialized<DecoderKeyValuePair>(
                    length: ConfigT.EncoderLayers);
            
            for (int i = 0; i < pastKeyValues.Length; i++)
            {
                pastKeyValues[i] = new();
            }
        }
        
        public void PopulateInputAndOutputWithPastKeyValuesAndSwapBuffers(
            List<NamedOnnxValue> inputs,
            List<NamedOnnxValue> outputs,
            int currentStepIndex)
        {
            var pastKeyValues = PastKeyValues;
            var pastKeyValuesNames = PastKeyValueNames<ConfigT>.NAMES;
            
            // https://imgur.com/a/florence2-decoder-merged-tI0RFxq

            // // [ batch_size, 16, encoder_sequence_length, 64 ]
            // ReadOnlySpan<int> encoderDimensions = [ 1, unchecked((int) ConfigT.EncoderLayers), encoderSequenceLength, 64 ];
            
            // [ batch_size, 16, decoder_sequence_length, 64 ]
            var decoderAttentionHeads = unchecked((int) ConfigT.DecoderAttentionHeads);
            
            // currentStepIndex is 1-based, but the input is based on previous number of InputIDs.
            // So we do currentStepIndex - 1
            
            ReadOnlySpan<int> decoderInputDimensions = [ 1, decoderAttentionHeads, currentStepIndex - 1, 64 ];
            
            ReadOnlySpan<int> decoderOutputDimensions = [ 1, decoderAttentionHeads, currentStepIndex, 64 ];

            var decoderInputLinearLength = decoderInputDimensions.GetDimensionSize();
            var decoderOutputLinearLength = decoderOutputDimensions.GetDimensionSize();
            
            for (int i = 0; i < ConfigT.EncoderLayers; i++)
            {
                // This MUST be passed via ref, as we perform swap operation below.
                ref var pastKeyValue = ref pastKeyValues[i];
                // We only use like half the names, pass via ref
                ref var pastKeyValueNames = ref pastKeyValuesNames[i];
                
                ref var pastKey = ref pastKeyValue.PastKey;
                ref var pastValue = ref pastKeyValue.PastValue;
                
                ref var presentKey = ref pastKeyValue.PresentKey;
                ref var presentValue = ref pastKeyValue.PresentValue;
                
                var pastKeyTensor = new DenseTensor<float>(
                    pastKey.AsMemory(0, decoderInputLinearLength),
                    decoderInputDimensions);
                
                var pastValueTensor = new DenseTensor<float>(
                    pastValue.AsMemory(0, decoderInputLinearLength),
                    decoderInputDimensions);
                
                var presentKeyTensor = new DenseTensor<float>(
                    presentKey.AsMemory(0, decoderOutputLinearLength),
                    decoderOutputDimensions);
                
                var presentValueTensor = new DenseTensor<float>(
                    presentValue.AsMemory(0, decoderOutputLinearLength),
                    decoderOutputDimensions);
                
                inputs.Add(pastKeyTensor.AsNamedOnnxValue(pastKeyValueNames.PastDecoderKeyName));
                inputs.Add(pastValueTensor.AsNamedOnnxValue(pastKeyValueNames.PastDecoderValueName));
                
                outputs.Add(presentKeyTensor.AsNamedOnnxValue(pastKeyValueNames.PresentDecoderKeyName));
                outputs.Add(presentValueTensor.AsNamedOnnxValue(pastKeyValueNames.PresentDecoderValueName));
                
                // Swap inputs with outputs! New outputs will become inputs, which avoids copying!
                
                (pastKey, presentKey) = (presentKey, pastKey);
                (pastValue, presentValue) = (presentValue, pastValue);
            }
        }
        
        public void CopyTo(
            in DecoderPastKeyValuesCollection<ConfigT> other,
            int currentStepIndex)
        {
            var pastKeyValues = PastKeyValues;
            var otherPastKeyValues = other.PastKeyValues;

            for (int i = 0; i < ConfigT.EncoderLayers; i++)
            {
                ref var pastKeyValue = ref pastKeyValues[i];
                ref var otherPastKeyValue = ref otherPastKeyValues[i];

                pastKeyValue.PastKey
                    .AsSpan(0, currentStepIndex)
                    .CopyTo(otherPastKeyValue.PastKey);

                pastKeyValue.PastValue
                    .AsSpan(0, currentStepIndex)
                    .CopyTo(otherPastKeyValue.PastValue);

                pastKeyValue.PresentKey
                    .AsSpan(0, currentStepIndex)
                    .CopyTo(otherPastKeyValue.PresentKey);
                
                pastKeyValue.PresentValue
                    .AsSpan(0, currentStepIndex)
                    .CopyTo(otherPastKeyValue.PresentValue);
            }
        }
    }

    public readonly struct EncoderPastKeyValuesCollection<ConfigT> where ConfigT : IFlorenceGenerationConfiguration
    {
        private struct EncoderKeyValuePair
        {
            public readonly float[] PastKey, PastValue;

            public EncoderKeyValuePair()
            {
                var size = PastKeyValuesHelpers<ConfigT>.ComputeTensorSizeFast(unchecked((int) ConfigT.MaxLength));

                // Negative infinity is just for debugging purposes -
                // It indicates that a given slot have yet to be written to.
                var fillValue = float.NegativeInfinity;

                PastKey = AllocationHelpers.AllocatePinnedArrayAndFill(size, fillValue);
                PastValue = AllocationHelpers.AllocatePinnedArrayAndFill(size, fillValue);
            }
        }

        private readonly EncoderKeyValuePair[] PastKeyValues;

        public EncoderPastKeyValuesCollection()
        {
            var pastKeyValues = PastKeyValues = AllocationHelpers
                .AllocatePinnedArrayUninitialized<EncoderKeyValuePair>(
                    length: ConfigT.EncoderLayers);

            for (int i = 0; i < pastKeyValues.Length; i++)
            {
                pastKeyValues[i] = new();
            }
        }

        public void PopulateInputPastKeyValues(
            List<NamedOnnxValue> inputs,
            List<NamedOnnxValue> outputs,
            int encoderSequenceLength,
            bool useCacheBranch)
        {
            var pastKeyValues = PastKeyValues;
            var pastKeyValuesNames = PastKeyValueNames<ConfigT>.NAMES;

            // https://imgur.com/a/florence2-decoder-merged-tI0RFxq

            // [ batch_size, 16, encoder_sequence_length, 64 ]
            var encoderAttentionHeads = unchecked((int) ConfigT.EncoderAttentionHeads);
            
            // For initial step where we do NOT use the cache branch, the input sequence length is 0.
            // Subsequent steps we feed the output from the initial step, and omit output tensors for.
            var encoderInputSequenceLength = useCacheBranch ? encoderSequenceLength : 0;

            ReadOnlySpan<int> encoderInputDimensions = [ 1, encoderAttentionHeads, encoderInputSequenceLength, 64 ];
            
            ReadOnlySpan<int> encoderOutputDimensions = [ 1, encoderAttentionHeads, encoderSequenceLength, 64 ];
            
            var encoderInputLinearLength = encoderInputDimensions.GetDimensionSize();

            var encoderOutputLinearLength = encoderOutputDimensions.GetDimensionSize();
            
            for (int i = 0; i < ConfigT.EncoderLayers; i++)
            {
                var pastKeyValue = pastKeyValues[i];
                ref var pastKeyValueNames = ref pastKeyValuesNames[i];

                var pastKey = new DenseTensor<float>(
                    pastKeyValue.PastKey.AsMemory(0, encoderInputLinearLength), 
                    encoderInputDimensions);
                
                var pastValue = new DenseTensor<float>(
                    pastKeyValue.PastValue.AsMemory(0, encoderInputLinearLength), 
                    encoderInputDimensions);

                inputs.Add(pastKey.AsNamedOnnxValue(pastKeyValueNames.PastEncoderKeyName));
                inputs.Add(pastValue.AsNamedOnnxValue(pastKeyValueNames.PastEncoderValueName));

                // For the first step where we don't use cache branch, we need to collect
                // the present key and value tensors for encoder.
                if (!useCacheBranch)
                {
                    var presentKey = new DenseTensor<float>(
                        pastKeyValue.PastKey.AsMemory(0, encoderOutputLinearLength), 
                        encoderOutputDimensions);
                
                    var presentValue = new DenseTensor<float>(
                        pastKeyValue.PastValue.AsMemory(0, encoderOutputLinearLength), 
                        encoderOutputDimensions);

                    outputs.Add(presentKey.AsNamedOnnxValue(pastKeyValueNames.PresentEncoderKeyName));
                    outputs.Add(presentValue.AsNamedOnnxValue(pastKeyValueNames.PresentEncoderValueName));
                }
            }
        }
    }
}