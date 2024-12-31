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

### Notes

I had to patch the following problem in `optimum/quanto/tensor/weights/qbytes.py` line 78

```
            output = torch.ops.quanto.qbytes_mm(input.view(-1, in_features), other._data, other._scale)
=> 
            output = torch.ops.quanto.qbytes_mm(input.contiguous().view(-1, in_features), other._data, other._scale)
```