from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from safetensors.torch import save_file
import torch

# Paths to base model components
base_model_path = "/Volumes/KINGSTON/sana-1.6b-1024px"

# Load the base model
pipeline = DiffusionPipeline.from_pretrained(base_model_path, torch_dtype=torch.float16)

# Add the scheduler (optional, if not already part of the pipeline)
pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)

# Save the combined base model components
output_path = "/Volumes/KINGSTON/ComfyUI/models/checkpoints/sana-1.6b-1024px_checkpoint.safetensors"
state_dict = pipeline.state_dict()
save_file(state_dict, output_path)

print(f"Combined base model checkpoint saved at {output_path}")
