using System;
using System.Runtime.CompilerServices;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace FlorenceSharp.Tensor
{
    public readonly struct ManagedTensor<T> where T: unmanaged
    {
        public readonly T[] ValuesArr;
        
        public readonly SystemNumericsTensors.Tensor<T> SNTensor;
        
        // public readonly OrtValue OnnxORTValue;

        public readonly DenseTensor<T> OnnxDenseTensor;
        
        public ManagedTensor(TensorDimensions dimensions, bool initialize, bool pinned = false)
            :this(initialize ? 
                SystemNumericsTensor.Create<T>(dimensions, pinned) : 
                SystemNumericsTensor.CreateUninitialized<T>(dimensions, pinned),
                dimensions) { }

        public ManagedTensor(SystemNumericsTensors.Tensor<T> snTensor): this(snTensor, snTensor.Lengths) { }
        
        public ManagedTensor(SystemNumericsTensors.Tensor<T> snTensor, TensorDimensions dimensions)
        {
            SNTensor = snTensor;

            var arr = ValuesArr = GetValuesArray(snTensor);
            
            // OnnxORTValue = OrtValue.CreateTensorValueFromMemory<T>(pinnedMemory, dimensions.WidenDimensions());

            // Span overload doesn't wrap memory ( Unsurprisingly )
            OnnxDenseTensor = new(arr.AsMemory(), dimensions);
            
            return;
            
            [UnsafeAccessor(UnsafeAccessorKind.Field, Name = "_values")]
            static extern ref T[] GetValuesArray(SystemNumericsTensors.Tensor<T> tensor);
        }

        public static ManagedTensor<T> CopyFromDenseTensor(DenseTensor<T> tensor, bool pinned = false)
        {
            // Unfortunately it is impossible to wrap DenseTensor, since ManagedTensor support
            // System.Numerics.Tensors.Tensor<T> as well, which is backed by an actual array.
            
            // Potential solution for avoiding copies: 
            // Pre-allocate pinned ManagedTensor<T> and use it as output for InferenceSession.Run()
            // Downside is having to manually compute output dimensions.
            
            return SystemNumericsTensor.Create<T>(
                tensor.Buffer.ToArray(), 
                (TensorDimensions) tensor.Dimensions, 
                pinned);
        }
        
        public NamedOnnxValue AsNamedOnnxValue(string name)
        {
            return NamedOnnxValue.CreateFromTensor(name, OnnxDenseTensor);
        }
        
        public static implicit operator SystemNumericsTensors.Tensor<T>(ManagedTensor<T> tensor)
        {
            return tensor.SNTensor;
        }
        
        public static implicit operator DenseTensor<T>(ManagedTensor<T> tensor)
        {
            return tensor.OnnxDenseTensor;
        }
        
        public static implicit operator ManagedTensor<T>(SystemNumericsTensors.Tensor<T> tensor)
        {
            return new(tensor);
        }
    }
}