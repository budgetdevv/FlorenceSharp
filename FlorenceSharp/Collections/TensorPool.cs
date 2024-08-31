using System.Collections.Generic;
using System.Runtime.CompilerServices;
using FlorenceSharp.Tensor;

namespace FlorenceSharp.Collections
{
    internal readonly struct TensorPool<T> where T: unmanaged
    {
        private readonly Stack<ManagedTensor<T>> Pool;

        private readonly TensorDimensions TensorDimensions;

        public TensorPool(int initialTensors, TensorDimensions tensorDimensions)
        {
            var pool = Pool = new(initialTensors);
                
            // Pre-allocate tensors
            for (int i = 0; i < initialTensors; i++)
            {
                pool.Push(AllocateTensor());
            }
                
            TensorDimensions = tensorDimensions;
        }
            
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public ManagedTensor<T> Take()
        {
            var pool = Pool;
                
            if (pool.TryPop(out var tensor))
            {
                return tensor;
            }
                
            return AllocateTensor();
        }
            
        [MethodImpl(MethodImplOptions.NoInlining)]
        private ManagedTensor<T> AllocateTensor()
        {
            return new(TensorDimensions, initialize: false);
        }
            
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void Return(ManagedTensor<T> tensor)
        {
            Pool.Push(tensor);
        }
    }
}