# Training an AI image generation model with your likeness

## Configuring SimpleTuner

Read the [SimpleTuner](https://github.com/bghira/SimpleTuner/) README and tutorial to learn how you can train an AI 
model to create images of a specific person.

I used 17 images like this in [examples/mikrei-new-training/](examples/mikrei-new-training/) to train a LyCORIS LoRA on top of the `terminusresearch/sana-1.6b-1024px` model.

![mikko12.jpg](https://raw.githubusercontent.com/mtreinik/ai-image-generation/refs/heads/main/examples/mikrei-new-training/mikko12.jpg)

Here are my configuration files:

- config.json
```
{
    "--resume_from_checkpoint": "latest",
    "--data_backend_config": "config/multidatabackend.json",
    "--aspect_bucket_rounding": 2,
    "--seed": 42,
    "--minimum_image_size": 0,
    "--output_dir": "output/sana-lora-mikrei-3e-4",
    "--lora_type": "lycoris",
    "--lycoris_config": "config/lycoris_config.json",
    "--max_train_steps": 10000,
    "--num_train_epochs": 0,
    "--checkpointing_steps": 500,
    "--checkpoints_total_limit": 5,
    "--hub_model_id": "simpletuner-sana-lora-mikrei-3e-4",
    "--push_to_hub": "true",
    "--push_checkpoints_to_hub": "true",
    "--tracker_project_name": "simpletuner-lora-training",
    "--tracker_run_name": "sana-lora-mikrei-3e-4",
    "--report_to": "wandb",
    "--model_type": "lora",
    "--pretrained_model_name_or_path": "terminusresearch/sana-1.6b-1024px",
    "--model_family": "sana",
    "--train_batch_size": 1,
    "--gradient_checkpointing": "false",
    "--caption_dropout_probability": 0.1,
    "--resolution_type": "pixel_area",
    "--resolution": 1024,
    "--validation_seed": 42,
    "--validation_steps": 500,
    "--validation_resolution": "1024x1024,1280x768",
    "--validation_guidance": 3.0,
    "--use_ema": true,
    "--validation_guidance_rescale": "0.0",
    "--validation_num_inference_steps": "30",
    "--user_prompt_library": "config/user_prompt_library.json",
    "--mixed_precision": "bf16",
    "--optimizer": "optimi-stableadamw",
    "--base_model_precision": "int8-quanto",
    "--text_encoder_1_precision": "no_change",
    "--text_encoder_2_precision": "no_change",
    "--text_encoder_3_precision": "no_change",
    "--learning_rate": "3e-4",
    "--lr_scheduler": "polynomial",
    "--lr_warmup_steps": 100,
    "--validation_torch_compile": "false",
    "--disable_benchmark": "false",
    "--validation_guidance_skip_layers": [7, 8, 9],
    "--validation_guidance_skip_layers_start": 0.01,
    "--validation_guidance_skip_layers_stop": 0.2,
    "--validation_guidance_skip_scale": 2.8,
    "--validation_guidance": 4.0,
    "--flux_use_uniform_schedule": true,
    "--flux_schedule_auto_shift": true,
    "--flux_schedule_shift": 0
}
```

- lycoris_config.json

```
{
    "algo": "lokr",
    "multiplier": 1.0,
    "linear_dim": 10000,
    "linear_alpha": 1,
    "factor": 8,
    "apply_preset": {
        "target_module": [
            "Attention",
            "FeedForward"
        ],
        "module_algo_map": {
            "Attention": {
                "factor": 8
            },
            "FeedForward": {
                "factor": 4
            }
        }
    }
}
```

- multidatabackend.json
```
[
    {
        "id": "mikrei-data-512px",
        "type": "local",
        "instance_data_dir": "mikrei-new-training",
        "caption_strategy": "instanceprompt",
        "instance_prompt": "mikrei",
        "cache_dir_vae": "vaecache/mikrei",
        "repeats": 100,
        "crop": false,
        "resolution": 512,
        "resolution_type": "pixel_area",
        "minimum_image_size":  192
    },
    {
        "id": "text-embeds",
        "type": "local",
        "dataset_type": "text_embeds",
        "default": true,
        "cache_dir": "cache/text/sana/mikrei",
        "disabled": false,
        "write_batch_size": 128
    }
]
```

- user_prompt_library.json

```
{
    "eating_pizza_mikrei": "A photo of mikrei eating a delicious slice of pizza at a restaurant",
    "feeding_dinosaurs_mikrei": "A photo of mikrei feeding a dinousaur in a jungle with lush vegetation and cinematic lighting",
    "piloting_a_plane_mikrei": "professional portrait of serious mikrei in a cockpit with instruments piloting a luxurious private jet in dramatic weather",
    "royal_mikrei": "close-up portrait of mikrei on a throne in a magnificent palace wearing a crown and royal attire",
    "chef_mikrei": "a high-quality, detailed photograph of mikrei as a sous-chef, immersed in the art of culinary creation",
    "just_mikrei": "a lifelike and intimate portrait of mikrei, showcasing his unique personality and charm",
    "cinematic_mikrei": "a cinematic, visually stunning photo of mikrei, emphasizing his dramatic and captivating presence",
    "elegant_mikrei": "an elegant and timeless portrait of mikrei, exuding grace and sophistication",
    "adventurous_mikrei": "a dynamic and adventurous photo of mikrei, captured in an exciting, action-filled moment",
    "mysterious_mikrei": "a mysterious and enigmatic portrait of mikrei, shrouded in shadows and intrigue",
    "vintage_mikrei": "a vintage-style portrait of mikrei, evoking the charm and nostalgia of a bygone era",
    "artistic_mikrei": "an artistic and abstract representation of mikrei, blending creativity with visual storytelling",
    "futuristic_mikrei": "a futuristic and cutting-edge portrayal of mikrei, set against a backdrop of advanced technology",
    "woman": "a beautifully crafted portrait of a woman, highlighting her natural beauty and unique features",
    "man": "a powerful and striking portrait of a man, capturing his strength and character",
    "boy": "a playful and spirited portrait of a boy, capturing youthful energy and innocence",
    "girl": "a charming and vibrant portrait of a girl, emphasizing her bright personality and joy",
    "family": "a heartwarming and cohesive family portrait, showcasing the bonds and connections between loved ones"
}
```



## Creating a Virtual Machine for AI training on Google Cloud Platform

I used a Virtual Machine (VM) on Google Cloud Platform (GCP) for training the AI model.

- Go to https://console.cloud.google.com/compute/instances
- Create an VM instance with an `NVIDIA A100 40GB` GPU
  - The GPU is not available on all zones, I used `europe-west4` (Netherlands) 
  - You may need to request additional quota for running an A100 GPU.
- Choose operating system `Deep Learning on Linux` and version with `CUDA 12.4`
- Choose at least 100 GB of disk space
- Start your VM instance

## Setting up and running SimpleTuner on the VM instance

- Create an account on Weights and Biases (wandb) and fetch your wandb API key from https://wandb.ai/quickstart
- Create an account on HuggingFace and an account token for reading and writing: https://huggingface.co/settings/tokens
- Connect to your VM instance with ssh (replace zone, instance name and project name with the ones you are using):
```
gcloud compute ssh --zone europe-west4-a instance-20241217-182510" --project stable-diffusion-training
```
- Answer 'y' to question about installing NVIDIA drivers
- Start screen and clone the SimpleTuner repository
```
screen
git clone https://github.com/bghira/SimpleTuner.git
cd SimpleTuner
```

- Setup your wandb and huggingface tokens:
```
cat >wandb.token
<enter your wandb API key here>
^D
cat >huggingface.token
<enter your huggingface token here>
^D
```

- Setup Python and other required libraries:
```
curl https://pyenv.run | bash
<add recommended stuff to ~/.bashrc>
<restart shell>
sudo apt-get update
sudo apt-get install -y build-essential \
                        libssl-dev \
                        zlib1g-dev \
                        libbz2-dev \
                        libreadline-dev \
                        libsqlite3-dev \
                        libncursesw5-dev \
                        libffi-dev \
                        liblzma-dev \
                        uuid-dev \
                        python3-bz2file 
pyenv install "3.11.9"
pyenv global 3.11.9
pyenv versions
echo "export PATH=$HOME/.pyenv/versions/3.11.9/bin:"'$PATH' >> ~/.bashrc
<restart shell>
```

- Activate Python virtual environment and install Python dependencies
```
python3.11 -m venv .venv
source .venv/bin/activate
pip install -U poetry pip
poetry config virtualenvs.create false
poetry install
```

- Log in to wandb and huggingface
```
wandb login $(cat wandb.token)
huggingface-cli login --token $(cat huggingface.token)
```

- Set up plain ASCII terminal mode
```
export LANG=C
export LC_ALL=C
export PYTHONIOENCODING=ascii
```

- Configure accelerate
```
accelerate config
```
- Copy training images and configuration files from your local machine to the VM instance
```
# Run these on your local machine in SimpleTuner directory. Assumes everything is in the config subdirectory: 
rsync --partial --progress --recursive --rsh ./gcloud-compute-ssh.sh config "instance-20241217-182510:SimpleTuner/"
```
- Run the training:
```
time ./train.sh
```

