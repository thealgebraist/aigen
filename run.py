import torch
from diffusers import Flux2KleinPipeline
import os

device = "cuda"
dtype = torch.bfloat16

pipe = Flux2KleinPipeline.from_pretrained(
    "black-forest-labs/FLUX.2-klein-4B", 
    torch_dtype=dtype
)
pipe.to(device)

output_dir = "/workspace/aigen"
base_prompt = "closeups of uncommonly ugly people with asymmetric faces too much hair weirdly large or small noses or ears or mouths and eyes and hair"
num_images = 32
num_inference_steps = 4

for i in range(num_images):
    prompt = f"{base_prompt}"
    image = pipe(
        prompt=prompt,
        num_inference_steps=num_inference_steps,
    ).images[0]

    image_path = os.path.join(output_dir, f"ugly{i+1}.png")
    image.save(image_path)

