using System;
using System.Linq;
using System.Text;
using FlorenceSharp.Tokenizers;

namespace Playground
{
    internal static class Program
    {
        private static void Main(string[] args)
        {
            while (true)
            {
                LoopBody();
            }
        }

        private static void LoopBody()
        {
            Console.Write("Input text to tokenize:");
            
            var sentences = Console.ReadLine()!.Split('|');
            
            var tokenizer = new FlorenceBartTokenizer(new());
            
            var output = tokenizer.Tokenize(sentences);

            var inputIDs = output.InputIDs.ToArray();
            
            var text =
            $"""
            Input IDs: {inputIDs.GetLongArrPrintString()}
            
            Attention Mask: {output.AttentionMask.ToArray().GetLongArrPrintString()}
            
            Decoded Text: {tokenizer.Decode(inputIDs)}
            """;
            
            Console.WriteLine(text);
        }

        private static string GetLongArrPrintString(this long[] arr)
        {
            // It has to include commas and the array brackets as well...
            var stringBuilder = new StringBuilder(arr.Length * 2);

            stringBuilder.Append('[');
            
            const string SEPARATOR = ", ";
            
            foreach (var item in arr)
            {
                stringBuilder.Append(item);
                stringBuilder.Append(SEPARATOR);
            }
            
            var separatorLength = SEPARATOR.Length;
            stringBuilder.Remove(stringBuilder.Length - separatorLength, separatorLength);
            
            stringBuilder.Append(']');
            
            return stringBuilder.ToString();
        }
    }
}