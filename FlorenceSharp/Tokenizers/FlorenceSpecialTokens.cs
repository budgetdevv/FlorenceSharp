namespace FlorenceSharp.Tokenizers
{
    public static class FlorenceSpecialTokens
    {
        // https://huggingface.co/TrumpMcDonaldz/Florence-2-large-onnx/blob/main/special_tokens_map.json#L7172

        // "bos_token": "<s>",
        // "cls_token": "<s>",
        // "eos_token": "</s>",
        // "mask_token": {
        //     "content": "<mask>",
        //     "lstrip": true,
        //     "normalized": true,
        //     "rstrip": false,
        //     "single_word": false
        // },
        // "pad_token": "<pad>",
        // "sep_token": "</s>",
        // "unk_token": "<unk>"
        
        public const string
            BEGINNING_OF_SEQUENCE = "<s>",
            CLASSIFICATION = BEGINNING_OF_SEQUENCE,
            END_OF_SEQUENCE = "</s>",
            SEPARATION = END_OF_SEQUENCE,
            PAD = "<pad>",
            UNKNOWN = "<unk>",
            MASK = "<mask>";
    }
}