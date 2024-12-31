# Generating images with the trained model

## generate-sana-lyroris-images.py

This repo contains the program `generate-sana-lyroris-images.py` for generating images based on the base model and the lora weights.

Run the following commands to generate an image of a sailor 
```
python -m venv venv
source venv/bin/activate    # On Linux/Mac
pip install -r requirements.txt
python generate-sana-lyroris-images.py 
```

Run the following command to see what parameters you can give to the program:
```
python generate-sana-lyroris-images.py -h
```

## Output

The directory [examples/](examples/) contains a few examples of generated images like this:

![2024-12-30-12-36-26-vintage_mikrei-037.jpg](https://raw.githubusercontent.com/mtreinik/ai-image-generation/refs/heads/main/examples/2024-12-30-12-36-26-vintage_mikrei-037.jpg)
![2024-12-30-12-38-55-futuristic_mikrei-041.jpg](https://raw.githubusercontent.com/mtreinik/ai-image-generation/refs/heads/main/examples/2024-12-30-12-38-55-futuristic_mikrei-041.jpg)

## Troubleshooting

If you have any problems with image generation, you can first try out the simpler image generation program that does not apply the LoRA adapter:

```
python generate-sana-image.py
```

It should generate an image like this:

![2024-12-31-09-07-58.png](https://raw.githubusercontent.com/mtreinik/ai-image-generation/refs/heads/main/examples/2024-12-31-09-07-58.png)

### RuntimeError 

I got the following error:

```
  File "/Users/mtreinik/miniconda3/lib/python3.12/site-packages/optimum/quanto/tensor/weights/qbytes.py", line 78, in forward
    output = torch.ops.quanto.qbytes_mm(input.view(-1, in_features), other._data, other._scale)
                                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: view size is not compatible with input tensor's size and stride (at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead.
```

To fix the problem I had to patch line 78 in  `optimum/quanto/tensor/weights/qbytes.py`:

```
            output = torch.ops.quanto.qbytes_mm(input.view(-1, in_features), other._data, other._scale)
=> 
            output = torch.ops.quanto.qbytes_mm(input.contiguous().view(-1, in_features), other._data, other._scale)
```