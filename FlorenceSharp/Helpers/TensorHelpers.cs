using System;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace FlorenceSharp.Helpers
{
    public static class TensorHelpers
    {
        // TODO: Write actual implementation instead of marshalling to System.Numerics.Tensors
        
        private static SystemNumericsTensors.Tensor<T> ToSystemNumericsTensor<T>(this DenseTensor<T> tensor)
        {
            var dimensionsWidened = tensor.Dimensions
                .ToArray()
                .Select(x => (nint) x)
                .ToArray()
                .AsSpan();
            
            return SystemNumericsTensor.Create(tensor.Buffer.ToArray(), dimensionsWidened);
        }
        
        private static DenseTensor<T> ToOnnxDenseTensor<T>(this SystemNumericsTensors.Tensor<T> tensor)
        {
            var dimensionsNarrowed = tensor.Lengths
                .ToArray()
                .Select(x => (int) x)
                .ToArray()
                .AsSpan();
            
            return new(tensor.ToArray(), dimensionsNarrowed);
        }
        
        
        // https://source.dot.net/#System.Numerics.Tensors/System/Numerics/Tensors/netcore/TensorExtensions.cs,268
        
        /// <summary>
        /// Join a sequence of tensors along an existing axis.
        /// </summary>
        /// <param name="tensors">The tensors must have the same shape, except in the dimension corresponding to axis (the first, by default).</param>
        public static DenseTensor<T> Concatenate<T>(params scoped ReadOnlySpan<DenseTensor<T>> tensors)
        {
            var convertedTensors = new SystemNumericsTensors.Tensor<T>[tensors.Length];

            var currentIndex = 0;
            
            foreach (var tensor in tensors)
            {
                convertedTensors[currentIndex++] = tensor.ToSystemNumericsTensor();
            }
            
            var concatenated = SystemNumericsTensor.Concatenate<T>(convertedTensors);

            return concatenated.ToOnnxDenseTensor();
        }

        /// <summary>
        /// Join a sequence of tensors along an existing axis.
        /// </summary>
        /// <param name="tensors">The tensors must have the same shape, except in the dimension corresponding to axis (the first, by default).</param>
        /// <param name="axis">The axis along which the tensors will be joined. If axis is -1, arrays are flattened before use. Default is 0.</param>
        public static DenseTensor<T> ConcatenateOnDimension<T>(
            int axis,
            params scoped ReadOnlySpan<DenseTensor<T>> tensors)
        {
            var convertedTensors = new SystemNumericsTensors.Tensor<T>[tensors.Length];

            var currentIndex = 0;
            
            foreach (var tensor in tensors)
            {
                convertedTensors[currentIndex++] = tensor.ToSystemNumericsTensor();
            }
            
            var concatenated = SystemNumericsTensor.ConcatenateOnDimension<T>(axis, convertedTensors);

            return concatenated.ToOnnxDenseTensor();
        }
        
        public static int GetTotalSizeForDimension(ReadOnlySpan<int> dimensions)
        {
            var totalSize = 1;
            
            foreach (var dimension in dimensions)
            {
                totalSize *= dimension;
            }
            
            return totalSize;
        }

        public static DenseTensor<T> CreateAndFillTensor<T>(T fill, ReadOnlySpan<int> dimensions)
        {
            var totalSize = GetTotalSizeForDimension(dimensions);

            var tensor = new DenseTensor<T>(totalSize);
            
            tensor.Fill(fill);
            
            return tensor;
        }
    }
}