from huggingface_hub import snapshot_download

# Specify the model repository and local path
model_repo = "terminusresearch/sana-1.6b-1024px"
local_dir = "/Volumes/KINGSTON/sana-1.6b-1024px"

# Download the entire repository
snapshot_download(repo_id=model_repo, local_dir=local_dir)