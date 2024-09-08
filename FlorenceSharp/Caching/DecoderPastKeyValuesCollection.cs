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
    
    public readonly struct DecoderPastKeyValuesCollection<ConfigT> where ConfigT: IFlorenceGenerationConfiguration
    {
        private struct DecoderKeyValuePair
        {
            public readonly float[]
                // These are for inputting
                PastKey,
                PastValue,
                // These are for outputting
                // PresentKeyValuesEncoderKey,
                // PresentKeyValuesEncoderValue,
                PresentKey,
                PresentValue;

            public DecoderKeyValuePair()
            {
                var size = ComputeTensorSizeFast(unchecked((int) ConfigT.MaxLength));
                
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

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static int ComputeTensorSizeFast(int sequenceLength)
        {
            // JIT should fold 1, ConfigT.EncoderLayers and 64 into a constant,
            // which means a single multiplication.
            
            // [ batch_size, 16, encoder_sequence_length, 64 ]
            // https://imgur.com/a/florence2-decoder-merged-tI0RFxq
            
            return unchecked((int) (1 * ConfigT.EncoderLayers * sequenceLength * 64));
        }
        
        // private static TensorDimensionInt4 ComputeTensorDimensions(int sequenceLength)
        // {
        //     // [ batch_size, 16, encoder_sequence_length, 64 ]
        //     // https://imgur.com/a/florence2-decoder-merged-tI0RFxq
        //
        //     return new(1, unchecked((int) ConfigT.EncoderLayers), sequenceLength, 64);
        // }
        
        public void PopulateInputAndOutputWithPastKeyValues(
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
            var decoderLayers = unchecked((int) ConfigT.DecoderLayers);
            
            // currentStepIndex is 1-based, but the input is based on previous number of InputIDs.
            // So we do currentStepIndex - 1
            
            ReadOnlySpan<int> decoderInputDimensions = [ 1, decoderLayers, currentStepIndex - 1, 64 ];
            
            ReadOnlySpan<int> decoderOutputDimensions = [ 1, decoderLayers, currentStepIndex, 64 ];

            var decoderInputLinearLength = decoderInputDimensions.GetDimensionSize();
            var decoderOutputLinearLength = decoderOutputDimensions.GetDimensionSize();
            
            for (int i = 0; i < ConfigT.EncoderLayers; i++)
            {
                var pastKeyValue = pastKeyValues[i];
                var pastKeyValueNames = pastKeyValuesNames[i];
                
                var pastKey = new DenseTensor<float>(
                    pastKeyValue.PastKey.AsMemory(0, decoderInputLinearLength),
                    decoderInputDimensions);
                
                var pastValue = new DenseTensor<float>(
                    pastKeyValue.PastValue.AsMemory(0, decoderInputLinearLength),
                    decoderInputDimensions);
                
                var presentKey = new DenseTensor<float>(
                    pastKeyValue.PresentKey.AsMemory(0, decoderOutputLinearLength),
                    decoderOutputDimensions);
                
                var presentValue = new DenseTensor<float>(
                    pastKeyValue.PresentValue.AsMemory(0, decoderOutputLinearLength),
                    decoderOutputDimensions);
                
                inputs.Add(pastKey.AsNamedOnnxValue(pastKeyValueNames.PastDecoderKeyName));
                inputs.Add(pastValue.AsNamedOnnxValue(pastKeyValueNames.PastDecoderValueName));
                
                outputs.Add(presentKey.AsNamedOnnxValue(pastKeyValueNames.PresentDecoderKeyName));
                outputs.Add(presentValue.AsNamedOnnxValue(pastKeyValueNames.PresentDecoderValueName));
            }
        }
    }

    public readonly struct EncoderPastKeyValuesCollection<ConfigT> where ConfigT : IFlorenceGenerationConfiguration
    {
        private struct EncoderKeyValuePair
        {
            public readonly float[]
                // These are for inputting
                PastKey,
                PastValue;

            public EncoderKeyValuePair()
            {
                var size = ComputeTensorSizeFast(unchecked((int)ConfigT.MaxLength));

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

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static int ComputeTensorSizeFast(int sequenceLength)
        {
            // JIT should fold 1, ConfigT.EncoderLayers and 64 into a constant,
            // which means a single multiplication.

            // [ batch_size, 16, decoder_sequence_length, 64 ]
            // https://imgur.com/a/florence2-decoder-merged-tI0RFxq

            return unchecked((int)(1 * ConfigT.EncoderLayers * sequenceLength * 64));
        }

        public void PopulateInputPastKeyValues(
            List<NamedOnnxValue> inputs,
            int encoderSequenceLength)
        {
            var pastKeyValues = PastKeyValues;
            var pastKeyValuesNames = PastKeyValueNames<ConfigT>.NAMES;

            // https://imgur.com/a/florence2-decoder-merged-tI0RFxq

            // [ batch_size, 16, encoder_sequence_length, 64 ]
            var encoderLayers = unchecked((int) ConfigT.DecoderLayers);

            ReadOnlySpan<int> encoderDimensions = [1, encoderLayers, encoderSequenceLength, 64];
            
            var encoderLinearLength = encoderDimensions.GetDimensionSize();

            for (int i = 0; i < ConfigT.EncoderLayers; i++)
            {
                var pastKeyValue = pastKeyValues[i];
                var pastKeyValueNames = pastKeyValuesNames[i];

                var pastKey = new DenseTensor<float>(
                    pastKeyValue.PastKey.AsMemory(0, encoderLinearLength), 
                    encoderDimensions);
                
                var pastValue = new DenseTensor<float>(
                    pastKeyValue.PastValue.AsMemory(0, encoderLinearLength), 
                    encoderDimensions);

                inputs.Add(pastKey.AsNamedOnnxValue(pastKeyValueNames.PastEncoderKeyName));
                inputs.Add(pastValue.AsNamedOnnxValue(pastKeyValueNames.PastEncoderValueName));
            }
        }
        
        public void PopulateOutputPresentKeyValues(
            List<NamedOnnxValue> outputs,
            int encoderSequenceLength)
        {
            var pastKeyValues = PastKeyValues;
            var pastKeyValuesNames = PastKeyValueNames<ConfigT>.NAMES;

            // https://imgur.com/a/florence2-decoder-merged-tI0RFxq

            // [ batch_size, 16, encoder_sequence_length, 64 ]
            var encoder = unchecked((int) ConfigT.DecoderLayers);

            ReadOnlySpan<int> encoderDimensions = [1, encoder, encoderSequenceLength, 64];
            
            var encoderLinearLength = encoderDimensions.GetDimensionSize();

            for (int i = 0; i < ConfigT.EncoderLayers; i++)
            {
                var pastKeyValue = pastKeyValues[i];
                var pastKeyValueNames = pastKeyValuesNames[i];

                var pastKey = new DenseTensor<float>(
                    pastKeyValue.PastKey.AsMemory(0, encoderLinearLength), 
                    encoderDimensions);
                
                var pastValue = new DenseTensor<float>(
                    pastKeyValue.PastValue.AsMemory(0, encoderLinearLength), 
                    encoderDimensions);

                outputs.Add(pastKey.AsNamedOnnxValue(pastKeyValueNames.PresentEncoderKeyName));
                outputs.Add(pastValue.AsNamedOnnxValue(pastKeyValueNames.PresentEncoderValueName));
            }
        }
    }
}