import torch
import datetime
from diffusers import SanaPipeline

prompt = "a lifelike and intimate portrait of a sailor, showcasing his unique personality and charm"
negative_prompt = "blurry, cropped, ugly"

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

model_id = 'terminusresearch/sana-1.6b-1024px'
pipeline = SanaPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=False
)
pipeline.to(device)

# Generate image
print("Generating image...")
image = pipeline(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=10,
    generator=torch.Generator(device=device).manual_seed(42),
    width=1024,
    height=1024,
    guidance_scale=4.0,
#    guidance_rescale=0.0,
).images[0]

# Save image
now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
image.save(f"{now}.png", format="PNG")
