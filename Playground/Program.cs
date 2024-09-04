using System;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
using System.Net.Http;
using System.Threading.Tasks;
using FlorenceSharp;
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

        private static async Task ImageCaptioningTest()
        {
            var imageBytes = await DownloadImageFromURL("https://i.imgur.com/drGJSNH.jpeg");
            
            var florence2 = new Florence2();

            Console.WriteLine(florence2.GenerateDetailedCaption(imageBytes));
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