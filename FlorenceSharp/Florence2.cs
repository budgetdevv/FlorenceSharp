using System;
using System.Collections.Frozen;
using System.Collections.Generic;
using System.Threading.Tasks;
using FlorenceSharp.Configs;
using FlorenceSharp.DecodingStrategies;
using FlorenceSharp.Helpers;
using FlorenceSharp.Processors.Imaging;
using FlorenceSharp.Processors.Logits;
using FlorenceSharp.StoppingCriteria;
using FlorenceSharp.Tensor;
using FlorenceSharp.Tokenizers;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace FlorenceSharp
{
    public struct DefaultFlorence2Config: IDefaultFlorence2Config;

    public sealed class Florence2(SessionOptions? onnxSessionOptions = null):
        Florence2<DefaultFlorence2Config>(onnxSessionOptions);

    public partial class Florence2<ConfigT>
    {
        private struct EncoderInput
        {
            // https://imgur.com/a/wtqPvud
            
            public const string
                INPUT_EMBEDS_NAME = "input_embeds",
                ATTENTION_MASK_NAME = "attention_mask";
        }
        
        private struct DecoderInput
        {
            // https://imgur.com/a/iVRFiGv
            
            public const string ENCODER_HIDDEN_STATES_NAME = "encoder_hidden_states";
        }
        
        private struct VisionEncoderInput
        {
            public const string PIXEL_VALUES_NAME = "pixel_values";
        }
        
        private struct TokenEmbeddingsInput
        {
            public const string INPUT_IDS_NAME = "input_ids";
        }

        private struct TokenEmbeddingsOutput
        {
            public const string INPUT_EMBEDS_NAME = "input_embeds";
        }
    }

    public partial class Florence2<ConfigT> : IAsyncInitializable<SessionOptions?, Florence2<ConfigT>>, IDisposable
        where ConfigT : struct, IFlorenceConfiguration
    {
        // https://huggingface.co/microsoft/Florence-2-large/blob/6bf179230dd8855083a51a5e11beb04aec1291fd/processing_florence2.py#L112
        private static readonly FrozenDictionary<FlorenceMode, string> PROMPTS_WITHOUT_INPUTS =
            new Dictionary<FlorenceMode, string>()
            {
                { FlorenceMode.Caption, "What does this image describe?" },
                { FlorenceMode.DetailedCaption, "Describe in detail what is shown in the image." },
                { FlorenceMode.MoreDetailedCaption, "Describe with a paragraph what is shown in the image." },
                { FlorenceMode.OCR, "What is the text in the image?" },
                { FlorenceMode.OCRWithRegion, "What is the text in the image, with regions?" },
                { FlorenceMode.ObjectDetection, "Locate the objects with category name in the image." },
                { FlorenceMode.DenseRegionCaption, "Locate the objects in the image, with their descriptions." },
                { FlorenceMode.RegionProposal, "Locate the region proposals in the image." },

            }.ToFrozenDictionary();

        private const string INPUT_PLACEHOLDER = "{input}";

        // https://huggingface.co/microsoft/Florence-2-large/blob/6bf179230dd8855083a51a5e11beb04aec1291fd/processing_florence2.py#L123
        private static readonly FrozenDictionary<FlorenceMode, string> PROMPTS_WITH_INPUTS =
            new Dictionary<FlorenceMode, string>()
            {
                { FlorenceMode.CaptionToPhraseGrounding, $"Locate the phrases in the caption: {INPUT_PLACEHOLDER}" },
                { FlorenceMode.ReferringExpressionSegmentation, $"Locate {INPUT_PLACEHOLDER} in the image with mask" },
                { FlorenceMode.RegionToSegmentation, $"What is the polygon mask of region {INPUT_PLACEHOLDER}" },
                { FlorenceMode.OpenVocabularyDetection, $"Locate {INPUT_PLACEHOLDER} in the image." },
                { FlorenceMode.RegionToCategory, $"What is the region {INPUT_PLACEHOLDER}?" },
                { FlorenceMode.RegionToDescription, $"What does the region {INPUT_PLACEHOLDER} describe?" },
                { FlorenceMode.RegionToOCR, $"What text is in the region {INPUT_PLACEHOLDER}?" },
            }.ToFrozenDictionary();

        private readonly SessionOptions OnnxSessionOptions;

        private readonly InferenceSession
            EncoderOnnxSession,
            DecoderOnnxSession,
            VisionEncoderOnnxSession,
            TokensEmbeddingOnnxSession;

        // https://huggingface.co/microsoft/Florence-2-large/blob/main/preprocessor_config.json
        private struct CLIPImageProcessorConfig : ICLIPImagePreProcessorConfig
        {
            public static int ImageWidth => 768;

            public static int ImageHeight => ImageWidth;

            public static int ImageSequenceLength => 577;

            public static float[] ImageMean => [0.485f, 0.456f, 0.406f];

            public static float[] ImageStd => [0.229f, 0.224f, 0.225f];
        }

        private readonly CLIPImagePreProcessor<CLIPImageProcessorConfig> ImagePreProcessor;

        private readonly FlorenceBartTokenizer Tokenizer;

        private readonly FlorenceLogitsProcessor LogitsProcessor;

        private readonly BeamSearcher<ConfigT> BeamSearcher;

        private readonly FlorenceStopCriteria<ConfigT> StoppingCriteria;

        protected Florence2(SessionOptions? onnxSessionOptions = null)
        {
            OnnxSessionOptions = onnxSessionOptions ??= new();

            EncoderOnnxSession = new(ConfigT.EncoderModelPath, OnnxSessionOptions);
            DecoderOnnxSession = new(ConfigT.DecoderModelPath, OnnxSessionOptions);
            VisionEncoderOnnxSession = new(ConfigT.VisionEncoderModelPath, OnnxSessionOptions);
            TokensEmbeddingOnnxSession = new(ConfigT.TokensEmbeddingModelPath, OnnxSessionOptions);

            ImagePreProcessor = new();

            Tokenizer = new(onnxSessionOptions);

            LogitsProcessor = new();

            BeamSearcher = new();

            StoppingCriteria = new();
        }

        public static async ValueTask<Florence2<ConfigT>> InitializeAsync(SessionOptions? onnxSessionOptions)
        {
            return new(onnxSessionOptions);
        }

        public string GenerateCaption(ReadOnlySpan<byte> imagePixels)
        {
            return GenerateCaptionCore(imagePixels, FlorenceMode.Caption);
        }

        public string GenerateDetailedCaption(ReadOnlySpan<byte> imagePixels)
        {
            return GenerateCaptionCore(imagePixels, FlorenceMode.DetailedCaption);
        }

        public string GenerateMoreDetailedCaption(ReadOnlySpan<byte> imagePixels)
        {
            return GenerateCaptionCore(imagePixels, FlorenceMode.MoreDetailedCaption);
        }

        internal ManagedTensor<float> GenerateEmbeddingsForInputIDs(DenseTensor<long> inputIDs)
        {
            var inputIDsOnnxValue = NamedOnnxValue.CreateFromTensor(TokenEmbeddingsInput.INPUT_IDS_NAME, inputIDs);
            
            return GenerateEmbeddingsForInputIDs(inputIDsOnnxValue, inputIDs.Dimensions);
        }
        
        internal ManagedTensor<float> GenerateEmbeddingsForInputIDs(Memory<long> inputIDs, int batchSize)
        {
            var tensor = new DenseTensor<long>(inputIDs, [ batchSize, inputIDs.Length / batchSize ]);
            
            return GenerateEmbeddingsForInputIDs(tensor);
        }
        
        internal ManagedTensor<float> GenerateEmbeddingsForInputIDs(NamedOnnxValue inputIDs, TensorDimensions dimensions)
        {
            var inputEmbeds = new ManagedTensor<float>(
                // [ batch_size, sequence_length, 1024 ]
                // 1024 is probably the size of the embeddings
                // inputIDs.Dimensions is batch_size, sequence_length ]
                dimensions: (ReadOnlySpan<nint>) [ ..(ReadOnlySpan<nint>) dimensions, 1024 ],
                initialize: true);
            
            // https://imgur.com/a/iyGFpRu
            // TODO: Does tokenized.InputIDs have the required dimensions?
            // Required Dimensions: [ batch_size, sequence_length ]
            TokensEmbeddingOnnxSession.Run(
                inputs: 
                [
                    inputIDs,
                ],
                outputs: 
                [
                    NamedOnnxValue.CreateFromTensor(TokenEmbeddingsOutput.INPUT_EMBEDS_NAME, inputEmbeds.OnnxDenseTensor),
                ]
            );

           return inputEmbeds;
        }

        internal ManagedTensor<float> DecodeIntoLogits(
            ManagedTensor<long> encoderAttentionMask,
            ManagedTensor<float> encoderHiddenStates,
            ManagedTensor<float> inputEmbeds)
        {
            throw new NotImplementedException();
        }
        
        private string GenerateCaptionCore(ReadOnlySpan<byte> imagePixels, FlorenceMode mode)
        {
            var prompt = PROMPTS_WITHOUT_INPUTS[mode];
            
            // TODO: We can make a cache for constant prompts.
            var tokenized = Tokenizer.Tokenize(prompt);
            
            var tokenEmbeddings = GenerateEmbeddingsForInputIDs(tokenized.InputIDs);
            
            var imagePreProcessed = ImagePreProcessor.PreProcess(imagePixels);

            var imagePixelsTensor = imagePreProcessed.NormalizedInputTensor;
            
            // The expected input is batch_size, channels ( Hardcoded to 3 ), height, width
            imagePixelsTensor.Reshape([ 1, ..imagePixelsTensor.Dimensions ]);
            
            using var visionEncoderOutput = VisionEncoderOnnxSession.Run(
            [
                NamedOnnxValue.CreateFromTensor(VisionEncoderInput.PIXEL_VALUES_NAME, imagePixelsTensor),
            ]);
            
            // TODO: Technically it is possible to supply a PINNED ( IMPORTANT ) ManagedTensor to .Run(), which would mean that
            // we can avoid copying! The downside is we have to calculate the resulting dimensions ourselves.
            var imageFeatures = ManagedTensor<float>
                .CopyFromDenseTensor((DenseTensor<float>) visionEncoderOutput[0].Value);
            
            var attentionMask = ManagedTensor<long>
                .CopyFromDenseTensor(tokenized.AttentionMask);
            
            var (mergedInputEmbeds, mergedAttentionMask) = 
                MergeTokenEmbeddingsAndImageFeatures(
                tokenEmbeddings,
                imageFeatures,
                attentionMask);
            
            // https://imgur.com/a/wtqPvud
            using var encoderOutput = EncoderOnnxSession.Run(
            [
                mergedInputEmbeds.AsNamedOnnxValue(EncoderInput.INPUT_EMBEDS_NAME),
                mergedAttentionMask.AsNamedOnnxValue(EncoderInput.ATTENTION_MASK_NAME),
            ]);
            
            var encoderHiddenStates = ManagedTensor<float>
                .CopyFromDenseTensor((DenseTensor<float>) encoderOutput[0].Value);
            
            return DecodeAndGenerateText(encoderHiddenStates, mergedAttentionMask);
        }

        private string DecodeAndGenerateText(
            ManagedTensor<float> encoderHiddenStates,
            ManagedTensor<long> mergedAttentionMask)
        {
            // // https://imgur.com/a/iVRFiGv
            // using var decoderOutput = DecoderOnnxSession.Run(
            // [
            //     encoderInputEmbedsOnnxValue,
            //     encoderAttentionMaskOnnxValue,
            //     NamedOnnxValue.CreateFromTensor(DecoderInput.ENCODER_HIDDEN_STATES_NAME, encoderHiddenStates),
            // ]);
            //
            // var logits = (DenseTensor<float>) decoderOutput[0].Value;
            
            // LogitsProcessor.ProcessLogits(logits);
            //
            // // Sample logits
            // // ...
            //
            // var response = Tokenizer.Decode(logits);

            ref readonly var beamSearcher = ref BeamSearcher;

            beamSearcher.Search(
                mergedAttentionMask,
                encoderHiddenStates,
                this,
                in Tokenizer,
                in StoppingCriteria);
            
            throw new NotImplementedException();
        }
        
        private static (ManagedTensor<float> mergedInputEmbeds, ManagedTensor<long> mergedAttentionMask) MergeTokenEmbeddingsAndImageFeatures(
            ManagedTensor<float> tokenEmbeddings,
            ManagedTensor<float> imageFeatures,
            ManagedTensor<long> textAttentionMask)
        {
            // var mergedInputEmbeds = TensorHelpers.ConcatenateOnDimension<float>(
            //     axis: 1, 
            //     tokenEmbeddings, 
            //     imageFeatures);
            //
            // var imageAttentionMask = TensorHelpers.CreateAndFillTensor(
            //     fill: 1L, 
            //     dimensions: imageFeatures.Dimensions[..2]);
            //
            // var mergedAttentionMask = TensorHelpers.ConcatenateOnDimension<long>(
            //     axis: 1, 
            //     textAttentionMask,
            //     imageAttentionMask);
            
            var mergedInputEmbeds = SystemNumericsTensor.ConcatenateOnDimension<float>(
                dimension: 1, 
                [tokenEmbeddings, 
                imageFeatures]);
            
            var imageAttentionMask = TensorHelpers.CreateAndFillTensor(
                fill: 1L, 
                dimensions: imageFeatures.OnnxDenseTensor.Dimensions[..2]);
            
            var mergedAttentionMask = SystemNumericsTensor.ConcatenateOnDimension<long>(
                dimension: 1, 
                [textAttentionMask,
                imageAttentionMask]);
            
            return (mergedInputEmbeds, mergedAttentionMask);
        }
        
        public void Dispose()
        {
            OnnxSessionOptions.Dispose();
            
            EncoderOnnxSession.Dispose();
            DecoderOnnxSession.Dispose();
            VisionEncoderOnnxSession.Dispose();
            TokensEmbeddingOnnxSession.Dispose();
            
            Tokenizer.Dispose();
        }
    }
}