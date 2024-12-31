from safetensors.torch import load_file

lora_adapter_path = "/Volumes/KINGSTON/ComfyUI/models/loras/pseudo-camera-pytorch_lora_weights.safetensors"

adapter = load_file(lora_adapter_path)
print(adapter.keys())