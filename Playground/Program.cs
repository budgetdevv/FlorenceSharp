using System;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
using System.Net.Http;
using System.Threading.Tasks;
using FlorenceSharp;
using FlorenceSharp.Configs;
using FlorenceSharp.Helpers;
using FlorenceSharp.Tokenizers;

namespace Playground
{
    internal static class Program
    {
        [Experimental("SYSLIB5001")]
        private static async Task Main(string[] args)
        {
            // TokenizerTest();

            await ImageCaptioningTest();
        }

        private static void TokenizerTest()
        {
            var tokenizer = new FlorenceBartTokenizer(new());

            while (true)
            {
                Console.Write("Input text to tokenize:");
            
                var sentences = Console.ReadLine()!.Split('|');
            
                var output = tokenizer.Tokenize(sentences);

                var inputIDs = output.InputIDs.ToArray();
                
                var text =
                $"""
                Input IDs: {inputIDs.GetArrPrintString()}

                Attention Mask: {output.AttentionMask.ToArray().GetArrPrintString()}

                Decoded Text: {tokenizer.Decode(inputIDs)}
                """;
            
                Console.WriteLine(text);
            }
        }

        private struct CUDAFlorenceConfig: IDefaultFlorence2Config
        {
            public static readonly DeviceType DEVICE_TYPE = DeviceType.CUDA;
            
            static ConfigurableOnnxModel.Configuration IFlorenceConfiguration.EncoderModelConfig
                => new ConfigurableOnnxModel.Configuration()
                    .WithDeviceType(DEVICE_TYPE)
                    .WithModelPath(IDefaultFlorence2Config.ENCODER_MODEL_PATH);
            
            static ConfigurableOnnxModel.Configuration IFlorenceConfiguration.DecoderModelConfig
                => new ConfigurableOnnxModel.Configuration()
                    .WithDeviceType(DEVICE_TYPE)
                    .WithModelPath(IDefaultFlorence2Config.DECODER_MODEL_PATH);
            
            static ConfigurableOnnxModel.Configuration IFlorenceConfiguration.VisionEncoderModelConfig
                => new ConfigurableOnnxModel.Configuration()
                    .WithDeviceType(DEVICE_TYPE)
                    .WithModelPath(IDefaultFlorence2Config.VISION_ENCODER_MODEL_PATH);
            
            static ConfigurableOnnxModel.Configuration IFlorenceConfiguration.TokensEmbeddingModelDeviceType
                => new ConfigurableOnnxModel.Configuration()
                    .WithDeviceType(DEVICE_TYPE)
                    .WithModelPath(IDefaultFlorence2Config.TOKENS_EMBEDDING_MODEL_PATH);
        }

        private static async Task ImageCaptioningTest()
        {
            var imageBytes = await DownloadImageFromURL("https://i.imgur.com/drGJSNH.jpeg");
            
            const bool USE_CUDA = false;

            if (USE_CUDA)
            {
                var florence2 = new Florence2<CUDAFlorenceConfig>();

                Console.WriteLine(florence2.GenerateMoreDetailedCaption(imageBytes));
            }

            else
            {
                var florence2 = new Florence2();

                Console.WriteLine(florence2.GenerateMoreDetailedCaption(imageBytes));
            }
        }
        
        private static async Task<byte[]?> DownloadImageFromURL(string url)
        {
            try
            {
                using var client = new HttpClient();
                
                return await client.GetByteArrayAsync(url);
            }
            
            catch (Exception ex)
            {
                Console.WriteLine($"Error downloading image: {ex.Message}");
                return null;
            }
        }
    }
}