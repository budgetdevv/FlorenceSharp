namespace FlorenceSharp
{
    public struct SentenceEncoderOutput
    {
        public long[] InputIDs;
        public long[] AttentionMask;

        public SentenceEncoderOutput(long[] inputIDs, long[] attentionMask)
        {
            InputIDs = inputIDs;
            AttentionMask = attentionMask;
        }
    }
}