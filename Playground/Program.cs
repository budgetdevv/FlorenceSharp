using System;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
using System.Net.Http;
using System.Threading.Tasks;
using FlorenceSharp;
using FlorenceSharp.Configs;
using FlorenceSharp.Helpers;
using FlorenceSharp.Tokenizers;
using Microsoft.ML.OnnxRuntime;

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

                Decoded Text: {tokenizer.Decode(inputIDs, skipSpecialTokens: false)}
                """;
            
                Console.WriteLine(text);
            }
        }

        private const bool USE_GPU = false;

        private static readonly OrtLoggingLevel LoggingLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_INFO;
        
        private struct FlorenceConfig: IDefaultFlorence2Config
        {
            public static readonly DeviceType DEVICE_TYPE = USE_GPU ? DeviceType.CUDA : DeviceType.CPU;
            
            static ConfigurableOnnxModel.Configuration IFlorenceConfiguration.EncoderModelConfig
                => new ConfigurableOnnxModel.Configuration()
                    .WithDeviceType(DEVICE_TYPE)
                    .WithLoggingLevel(LoggingLevel)
                    .WithModelPath(IDefaultFlorence2Config.ENCODER_MODEL_PATH);
            
            static ConfigurableOnnxModel.Configuration IFlorenceConfiguration.DecoderModelConfig
                => new ConfigurableOnnxModel.Configuration()
                    .WithDeviceType(DEVICE_TYPE)
                    .WithLoggingLevel(LoggingLevel)
                    .WithModelPath(IDefaultFlorence2Config.DECODER_MODEL_PATH);
            
            static ConfigurableOnnxModel.Configuration IFlorenceConfiguration.VisionEncoderModelConfig
                => new ConfigurableOnnxModel.Configuration()
                    .WithDeviceType(DEVICE_TYPE)
                    .WithLoggingLevel(LoggingLevel)
                    .WithModelPath(IDefaultFlorence2Config.VISION_ENCODER_MODEL_PATH);
            
            static ConfigurableOnnxModel.Configuration IFlorenceConfiguration.TokensEmbeddingModelDeviceType
                => new ConfigurableOnnxModel.Configuration()
                    .WithDeviceType(DEVICE_TYPE)
                    .WithLoggingLevel(LoggingLevel)
                    .WithModelPath(IDefaultFlorence2Config.TOKENS_EMBEDDING_MODEL_PATH);
        }

        private static async Task ImageCaptioningTest()
        {
            var imageBytes = await DownloadImageFromURL("https://avatars.githubusercontent.com/u/74057874?v=4");
            
            var florence2 = new Florence2<FlorenceConfig>();

            var sw = new Stopwatch();
            
            const int NUM_ITERATIONS = 3;
            
            for (int i = 0; i < NUM_ITERATIONS; i++)
            {
                sw.Start();
                
                var result = florence2.GenerateMoreDetailedCaption(imageBytes);

                sw.Stop();

                var elapsedSeconds = sw.Elapsed.TotalSeconds;
                
                sw.Reset();
                
                Console.WriteLine(
                $"""
                {result}
                
                --------------------------------
                
                Took {elapsedSeconds} seconds!
                """);
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