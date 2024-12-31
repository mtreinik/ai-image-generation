prompt = "a lifelike and intimate portrait of mikrei, showcasing his unique personality and charm"
negative_prompt = 'blurry, cropped, ugly'

import torch
import datetime
from diffusers import SanaPipeline
from lycoris import create_lycoris_from_weights

# def download_adapter(repo_id: str):
#     import os
#     from huggingface_hub import hf_hub_download
#     adapter_filename = "pytorch_lora_weights.safetensors"
#     cache_dir = os.environ.get('HF_PATH', os.path.expanduser('~/.cache/huggingface/hub/models'))
#     cleaned_adapter_path = repo_id.replace("/", "_").replace("\\", "_").replace(":", "_")
#     path_to_adapter = os.path.join(cache_dir, cleaned_adapter_path)
#     path_to_adapter_file = os.path.join(path_to_adapter, adapter_filename)
#     os.makedirs(path_to_adapter, exist_ok=True)
#     hf_hub_download(
#         repo_id=repo_id, filename=adapter_filename, local_dir=path_to_adapter
#     )
#
#     return path_to_adapter_file

model_id = 'terminusresearch/sana-1.6b-1024px'
# adapter_repo_id = 'mtreinik/simpletuner-sana-lora-mikrei-3e-4'
# adapter_filename = 'pytorch_lora_weights.safetensors'
# adapter_file_path = download_adapter(repo_id=adapter_repo_id)
#pipeline = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16, low_cpu_mem_usage=False, device_map=None) # loading directly in bf16
#pipeline = DiffusionPipeline.from_pretrained(model_id, low_cpu_mem_usage=False, device_map=None)
pipeline = SanaPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16, low_cpu_mem_usage=False, device_map=None)

# lora_scale = 1.0
# wrapper, _ = create_lycoris_from_weights(lora_scale, adapter_file_path, pipeline.transformer)
# wrapper.merge_to()

## Optional: quantise the model to save on vram.
## Note: The model was quantised during training, and so it is recommended to do the same during inference time.
from optimum.quanto import quantize, freeze, qint8
quantize(pipeline.transformer, weights=qint8)
freeze(pipeline.transformer)

pipeline.to('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu') # the pipeline is already in its target precision level
print("generating image")
image = pipeline(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=10,
    generator=torch.Generator(device='cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu').manual_seed(42),
    width=1024,
    height=1024,
    guidance_scale=4.0,
    guidance_rescale=0.0,
).images[0]

now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

image.save(f"output-{now}.png", format="PNG")