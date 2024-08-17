import onnx;
from transformers import AutoProcessor, BartTokenizer;
from onnxruntime_extensions import OrtPyFunction, gen_processing_models;

# BartTokenizer

# https://discuss.huggingface.co/t/bart-input-confusion/1103/2
TOKENIZER_MAX_LENGTH = 1024;

# My repo is a combination of microsoft/Florence-2-large and onnx-community/Florence-2-large
# microsoft/Florence-2-large lacked the required merges.txt,
# while onnx-community/Florence-2-large lacked processing_florence2.py
HF_REPO = "TrumpMcDonaldz/Florence-2-large-onnx";

processor = AutoProcessor.from_pretrained(
    HF_REPO,
    trust_remote_code=True,
    # force_download=True,
    # "ValueError: Please use the slow version of the tokenizer (ex: RobertaTokenizer)."
    # https://github.com/microsoft/onnxruntime-extensions/pull/521#pullrequestreview-1577708832
    use_fast=False);

tokenizer: BartTokenizer = processor.tokenizer;

print(type(tokenizer));


def generate_models():
    # pre_kwargs={}, post_kwargs={} tells it to generate both pre and post-processing onnx models for the tokenizer
    pre_processing_model, post_processing_model = gen_processing_models(tokenizer, pre_kwargs={}, post_kwargs={});

    # Export the ONNX model to a file
    onnx.save_model(OrtPyFunction(pre_processing_model).onnx_model, "florence2_tokenizer_encode.onnx");

    onnx.save_model(OrtPyFunction(post_processing_model).onnx_model, "florence2_tokenizer_decode.onnx");

    print("Model exported successfully!");


def test_encoding():
    while (True):
        output = tokenizer.encode(input("Input text to encode!"));
        print(output);


# Run any of the above functions
test_encoding();