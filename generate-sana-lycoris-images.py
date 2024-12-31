from dataclasses import dataclass, astuple
from diffusers import SanaPipeline
from huggingface_hub import hf_hub_download
from lycoris import create_lycoris_from_weights
from typing import Dict, Tuple
import argparse
import datetime
import json
import os
import torch

# Base model repo on HuggingFace
model_id = 'terminusresearch/sana-1.6b-1024px'

# LoRA adapter repo on HuggingFace
adapter_repo_id = 'mtreinik/simpletuner-sana-lora-mikrei-3e-4'

@dataclass
class Prompt:
    prompt_id: str
    prompt: str
    negative_prompt: str

@dataclass
class Parameters:
    count: int
    initial_seed: int
    prompts: Dict[str, Tuple[str, str]]
    output_dir: str
    steps: int

@dataclass
class Generator:
    device: str
    pipeline: SanaPipeline

def read_parameters():
    parser = argparse.ArgumentParser()

    parser.add_argument('--count', type=int, default=1, help="Image batch count: how many times an image should be generated")
    parser.add_argument('--seed', type=int, default=42, help="Initial seed value")
    parser.add_argument('--id', type=str, default="image", help="Image ID added to output file name")
    parser.add_argument('--prompt', type=str, default="man", help="Prompt for image generation")
    parser.add_argument('--negative_prompt', type=str, default="blurry, cropped, ugly", help="Negative prompt: avoid these in the generated image")
    parser.add_argument('--prompt_file', type=str, help="")
    parser.add_argument('--output_dir', type=str, default="output", help="directory for generated images")
    parser.add_argument('--steps', type=int, default=10, help="Number of steps to iterate")
    args = parser.parse_args()

    if args.prompt_file:
        with open(args.prompt_file, 'r') as file:
            prompts = json.load(file)
        print(f"Loaded {len(prompts)} prompts from file {args.prompt_file}")
    else:
        prompts = {args.id: (args.prompt, args.negative_prompt)}
        print(f"Using single prompt '{args.prompt}'")

    return Parameters(
        count=args.count,
        initial_seed=args.seed,
        prompts=prompts,
        output_dir=args.output_dir,
        steps=args.steps
    )

def download_adapter(repo_id: str) -> str:
    adapter_filename = "pytorch_lora_weights.safetensors"
    cache_dir = os.environ.get('HF_PATH', os.path.expanduser('~/.cache/huggingface/hub/models'))
    cleaned_adapter_path = repo_id.replace("/", "_").replace("\\", "_").replace(":", "_")
    path_to_adapter = os.path.join(cache_dir, cleaned_adapter_path)
    path_to_adapter_file = os.path.join(path_to_adapter, adapter_filename)
    os.makedirs(path_to_adapter, exist_ok=True)
    hf_hub_download(
        repo_id=repo_id, filename=adapter_filename, local_dir=path_to_adapter
    )

    return path_to_adapter_file

def quantize_pipeline(pipeline):
    from optimum.quanto import quantize, freeze, qint8
    quantize(pipeline.transformer, weights=qint8)
    freeze(pipeline.transformer)

def initialize_generator():
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    pipeline = SanaPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=False
    )
    lora_scale = 1.0
    adapter_file_path = download_adapter(adapter_repo_id)
    wrapper, _ = create_lycoris_from_weights(lora_scale, adapter_file_path, pipeline.transformer)
    wrapper.merge_to()

# this could save memory?
#    quantize_pipeline(pipeline)

    pipeline.to(device)
    return Generator(device=device, pipeline=pipeline)

def generate_images(parameters: Parameters, generator: Generator):
    device = generator.device
    pipeline = generator.pipeline
    count = parameters.count
    initial_seed = parameters.initial_seed
    steps = parameters.steps
    prompts = parameters.prompts
    output_dir = parameters.output_dir

    image_index = 1
    for prompt_index, (prompt_id, (prompt, negative_prompt)) in enumerate(prompts.items(), start=1):
        print(f"Prompt {prompt_index}/{len(prompts)} {prompt_id}: '{prompt}' with negative prompt '{negative_prompt}' ({steps} steps)")
        for iter in range(1, count+1):
            seed = initial_seed + iter - 1
            image = pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=steps,
                generator=torch.Generator(device=device).manual_seed(seed),
                width=1024,
                height=1024,
                guidance_scale=4.0,
            ).images[0]

            now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            output_file = f"{output_dir}/{now}-{prompt_index:03}-{prompt_id}-{image_index:03}.jpg"
            image.save(output_file, format="JPEG")
            print(f"Wrote image {iter}/{count} of {prompt_id} with seed {seed} at '{output_file}'")

            image_index = image_index + 1

if __name__ == "__main__":
    parameters = read_parameters()
    generator = initialize_generator()
    generate_images(parameters, generator)
