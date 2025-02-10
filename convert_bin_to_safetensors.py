# Convert bin to safetensors

import torch
from safetensors.torch import save_file

model = torch.load("models/teticio/audio-diffusion-256/unet/diffusion_pytorch_model.bin")

# Save as safetensors
save_file(model, "models/teticio/audio-diffusion-256/unet/diffusion_pytorch_model.safetensors")

