using System.Threading.Tasks;

namespace FlorenceSharp
{
    public interface IAsyncInitializable<ThisT> where ThisT: IAsyncInitializable<ThisT>
    {
        public static abstract ValueTask<ThisT> InitializeAsync();
    }
    
    public interface IAsyncInitializable<InputT, ThisT> where ThisT: IAsyncInitializable<InputT, ThisT>
    {
        public static abstract ValueTask<ThisT> InitializeAsync(InputT input);
    }
}