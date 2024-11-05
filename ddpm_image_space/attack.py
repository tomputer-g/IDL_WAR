from diffusers import DDPMPipeline
from PIL import Image
import numpy as np
import torch
import torchvision

DEVICE = 'cuda'

# load model and scheduler
pipe = DDPMPipeline.from_pretrained("google/ddpm-cifar10-32").to(DEVICE)

pil_image = Image.open("/home/rmdluo/IDL_WAR/zimingg_sandbox/watermarked.png")
image = torchvision.transforms.functional.pil_to_tensor(pil_image)
image = image.float() / 255.0
image = image.to(DEVICE)

print(image)

def clamp(image):
    image = image.clamp(0, 1)
    return image

timestep = torch.tensor([0])
timestep = timestep.to(DEVICE)
noise = torch.randn_like(image)
noise = noise.to(DEVICE)
noisy_image = pipe.scheduler.add_noise(image, noise, timestep)
noisy_image = clamp(noisy_image)
noisy_image = noisy_image.to(DEVICE)

print(noisy_image)

# save image
noisy_image_pil = torchvision.transforms.functional.to_pil_image(noisy_image)
noisy_image_pil.save("noisy_test.png")

with torch.no_grad():
    print(noisy_image.device)
    print(timestep.device)
    denoised_output = pipe.unet(torch.unsqueeze(noisy_image, 0), timestep).sample  # Forward pass through the model
    denoised_image = pipe.scheduler.step(denoised_output, timestep, noisy_image).prev_sample
    denoised_image = clamp(denoised_image)

print(denoised_image)
denoised_output_pil = torchvision.transforms.functional.to_pil_image(denoised_image[0])
denoised_output_pil.save("denoised_test.png")