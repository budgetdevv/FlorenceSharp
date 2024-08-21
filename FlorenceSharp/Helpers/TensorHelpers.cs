using System;
using System.Linq;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using FlorenceSharp.Tensor;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace FlorenceSharp.Helpers
{
    public static class TensorHelpers
    {
        // TODO: Write actual implementation instead of marshalling to System.Numerics.Tensors
        
        // https://source.dot.net/#System.Numerics.Tensors/System/Numerics/Tensors/netcore/TensorExtensions.cs,268
        
        public static ManagedTensor<T> CreateAndFillTensor<T>(T fill, ReadOnlySpan<int> dimensions)
            where T : unmanaged
        {
            var tensor = new ManagedTensor<T>(dimensions, initialize: false);
            
            tensor.SNTensor.Fill(fill);
            
            return tensor;
        }
        
        public static ManagedTensor<T> SoftMax<T>(this ManagedTensor<T> tensor)
            where T: unmanaged, IExponentialFunctions<T>
        {
            return SystemNumericsTensor.SoftMax<T>(tensor.SNTensor);
        }
        
        public static ManagedTensor<T> SoftMaxInPlace<T>(this ManagedTensor<T> tensor)
            where T: unmanaged, IExponentialFunctions<T>
        {
            var snTensor = tensor.SNTensor;
            
            SystemNumericsTensor.SoftMax<T>(snTensor, snTensor);
            
            return tensor;
        }
        
        private readonly struct TopKSession
        {
            public readonly InferenceSession Model;

            public readonly ManagedTensor<ulong> KInputBuffer;

            public TopKSession()
            {
                Model = new(
                    ResourceHelpers.GetResourceBytes(
                        typeof(TensorHelpers).Assembly, 
                        "TopK.onnx")!);

                KInputBuffer = new(
                    (ReadOnlySpan<nint>) [ 1 ], 
                    initialize: false,
                    pinned: true);
            }
        }
        
        [ThreadStatic]
        private static TopKSession? TopKSessionSessionThreadStatic;

        private static TopKSession TopKSessionSessionCurrentThread
        {
            get
            {
                return TopKSessionSessionThreadStatic ?? CreateAndSetTopK();

                [MethodImpl(MethodImplOptions.NoInlining)]
                TopKSession CreateAndSetTopK()
                {
                    return (TopKSessionSessionThreadStatic = new TopKSession()).GetValueOrDefault();
                }
            }
        }
        
        public readonly struct TopKOutput(ManagedTensor<float> logits, ManagedTensor<long> indices)
        {
            public readonly ManagedTensor<float> Logits = logits;
            
            public readonly ManagedTensor<long> Indices = indices;
        }
        
        public static TopKOutput TopK(this ManagedTensor<float> logitsInput, ulong k, bool pinned = false)
        {
            // https://josephrocca.github.io/onnxscript-editor/demo/
            
            // import onnx
            // from onnx import TensorProto
            // from onnx.helper import make_tensor
            //     from onnxscript import script, INT64, FLOAT
            // from onnxscript import opset18 as op
            //
            // @script(op)
            // def model(logits: FLOAT[...], k: INT64):
            // return op.TopK(logits, k);
            //
            // onnx.save(model.to_model_proto(), "/model.onnx");
            
            // https://github.com/onnx/onnx/blob/main/docs/Operators.md#TopK
            
            var dimensions = (TensorDimensions) logitsInput.SNTensor.Lengths;
            
            // Create new output buffers

            var logitsOutput = new ManagedTensor<float>(dimensions, initialize: false, pinned);
            
            var indicesOutput = new ManagedTensor<long>(dimensions, initialize: false, pinned);

            var topK = TopKSessionSessionCurrentThread;
            
            var topKModel = topK.Model;
            
            var kInputBuffer = topK.KInputBuffer;

            // Probably the fastest way to store its value
            MemoryMarshal.GetArrayDataReference(kInputBuffer.ValuesArr) = k;

            topKModel.Run(
                inputs: 
                [
                    NamedOnnxValue.CreateFromTensor<float>(
                        "logits", 
                        logitsInput),
                    
                    NamedOnnxValue.CreateFromTensor<ulong>(
                        "k",
                        kInputBuffer),
                ], 
                outputs:
                [ 
                    NamedOnnxValue.CreateFromTensor<float>(
                        "values",
                        logitsOutput),
                    
                    NamedOnnxValue.CreateFromTensor<long>(
                        "indices",
                        indicesOutput),
                ]
            );
            
            return new(logitsOutput, indicesOutput);
        }
        
        // Generated by ChatGPT
        public static void PrintTensor<T>(this DenseTensor<T> tensor)
        {
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