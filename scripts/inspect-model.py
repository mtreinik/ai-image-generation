import torch

model_path = "/Volumes/KINGSTON/ComfyUI/models/checkpoints/Sana_1600M_1024px.pth"

model_data = torch.load(model_path, map_location="cpu")

# Check if the file contains a state_dict
if isinstance(model_data, dict):
    if "state_dict" in model_data:  # Common pattern
        state_dict = model_data["state_dict"]
    else:
        state_dict = model_data  # Assume it's the state_dict itself

    # Print parameter names
    print(state_dict.keys())
else:
    print("Unexpected format:", type(model_data))