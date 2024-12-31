#from diffusers import SanaTransformer2DModel
#import torch
#transformer_path = "/Volumes/KINGSTON/sana-1.6b-1024px/transformer"
#transformer = SanaTransformer2DModel.from_pretrained(transformer_path, torch_dtype=torch.float16)
#print(transformer.state_dict().keys())

from safetensors.torch import load_file

transformer_checkpoint_path = "/Volumes/KINGSTON/sana-1.6b-1024px/transformer/diffusion_pytorch_model.safetensors"
model_weights = load_file(transformer_checkpoint_path)
print(model_weights.keys())