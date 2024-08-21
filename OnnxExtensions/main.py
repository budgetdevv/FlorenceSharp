import os

import onnx;
from unittest.mock import patch
from transformers import AutoModelForCausalLM, AutoProcessor, BartTokenizer
from transformers.dynamic_module_utils import get_imports
from onnxruntime_extensions import OrtPyFunction, gen_processing_models;
import requests
from PIL import Image;

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


def generate_tokenizer_models():
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


# https://huggingface.co/microsoft/Florence-2-large-ft/discussions/4

def fixed_get_imports(filename: str | os.PathLike) -> list[str]:
    """Work around for https://huggingface.co/microsoft/phi-1_5/discussions/72."""
    if not str(filename).endswith("/modeling_florence2.py"):
        return get_imports(filename);
    imports = get_imports(filename);
    imports.remove("flash_attn");
    return imports;


def get_device_type():
    import torch
    if torch.cuda.is_available():
        return "cuda";
    else:
        if (torch.backends.mps.is_available() and torch.backends.mps.is_built()):
            return "mps";
        else:
            return "cpu";


def run_florence2():
    import subprocess

    MODEL_NAME = "microsoft/Florence-2-base";

    TASK = "<MORE_DETAILED_CAPTION>";

    device = get_device_type();

    if (device == "cuda"):
        subprocess.run('pip install flash-attn --no-build-isolation',
                       env={'FLASH_ATTENTION_SKIP_CUDA_BUILD': "TRUE"},
                       shell=True)
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True);

    else:
        # https://huggingface.co/microsoft/Florence-2-base-ft/discussions/4
        with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports):
            model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True);

    model.to(device);

    # Set up the prompt and image
    prompt = TASK;
    url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg?download=true"
    image = Image.open(requests.get(url, stream=True).raw);

    # Process inputs
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device);

    # Generate output
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        num_beams=3,
        do_sample=False,
    ).to(device);

    # Decode and post-process
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0];

    parsed_answer = processor.post_process_generation(
        generated_text,
        task=TASK,
        image_size=(image.width, image.height));

    print(parsed_answer);


# Run any of the above functions

# generate_tokenizer_models();
# test_encoding();
run_florence2();