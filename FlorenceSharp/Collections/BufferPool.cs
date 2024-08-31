using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;

namespace FlorenceSharp.Collections
{
    internal readonly struct BufferPool<T> where T: unmanaged
    {
        private readonly Stack<T[]> Pool;

        private readonly int Size;

        public BufferPool(int initialBuffers, int size)
        {
            var pool = Pool = new(initialBuffers);
                
            // Pre-allocate tensors
            for (int i = 0; i < initialBuffers; i++)
            {
                pool.Push(AllocateBuffer());
            }

            Size = size;
        }
            
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public T[] Take()
        {
            var pool = Pool;
                
            if (pool.TryPop(out var tensor))
            {
                return tensor;
            }
                
            return AllocateBuffer();
        }
            
        [MethodImpl(MethodImplOptions.NoInlining)]
        private T[] AllocateBuffer()
        {
            return GC.AllocateUninitializedArray<T>(Size);
        }
            
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void Return(T[] buffer)
        {
            Pool.Push(buffer);
        }
    }
}