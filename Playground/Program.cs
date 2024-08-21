using System;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
using System.Text;
using FlorenceSharp.Helpers;
using FlorenceSharp.Tensor;
using FlorenceSharp.Tokenizers;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace Playground
{
    internal static class Program
    {
        [Experimental("SYSLIB5001")]
        private static void Main(string[] args)
        {
            var tokenizer = new FlorenceBartTokenizer(new());
            
            while (true)
            {
                LoopBody(tokenizer);
            }
        }

        private static void LoopBody(FlorenceBartTokenizer tokenizer)
        {
            Console.Write("Input text to tokenize:");
            
            var sentences = Console.ReadLine()!.Split('|');
            
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