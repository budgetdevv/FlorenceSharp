using System;
using System.Text;
using FlorenceSharp.Tensor;

namespace FlorenceSharp.Helpers
{
    public static class DebugHelpers
    {
        public static string GetSpanPrintString<T>(this ReadOnlySpan<T> span)
        {
            return span.ToArray().GetArrPrintString();
        }
        
        public static string GetArrPrintString<T>(this T[] arr)
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
        
        // Generated by ChatGPT
        public static void PrintTensor<T>(this ManagedTensor<T> managedTensor)
            where T: unmanaged
        {
            var tensor = managedTensor.OnnxDenseTensor;
            
            // Get the shape of the tensor
            var shape = string.Join("x", tensor.Dimensions.ToArray());
            
            // Print out the flat array
            Console.WriteLine($"Tensor Shape: {shape}");
            Console.WriteLine("Values:");
            
            for (int i = 0; i < tensor.Length; i++)
            {
                Console.Write(tensor.GetValue(i) + " ");
            }
            Console.WriteLine();
        }
    }
}