from diffusers import DDIMPipeline
from PIL import Image
import torch
import torchvision
from tqdm import tqdm
import os
import shutil

# upscale
from RealESRGAN import RealESRGAN

DEVICE = 'cuda'

# load model and scheduler
pipe = DDIMPipeline.from_pretrained("google/ddpm-cifar10-32").to(DEVICE)

size=512

blend = [(1, 0.25), (2, 0.75), (4, 0), (8, 0)]
sr_models = {
    2: RealESRGAN(DEVICE, scale=2),
    4: RealESRGAN(DEVICE, scale=4),
    8: RealESRGAN(DEVICE, scale=8),
}

for scale, model in sr_models.items():
    model.load_weights(f'weights/RealESRGAN_x{scale}.pth', download=True)

def clamp(image):
    image = image.clamp(0, 1)
    return image

def diffusion_attack(images, timesteps):
    pipe.scheduler.set_timesteps(timesteps)
    timestep = torch.tensor([timesteps])
    timestep = timestep.to(DEVICE)

    noise = torch.randn_like(images)
    noise = noise.to(DEVICE)
    noisy_images = pipe.scheduler.add_noise(images, noise, timestep)
    noisy_images = noisy_images.to(DEVICE)

    for t in reversed(range(timesteps)):
        t = torch.tensor([t], device=DEVICE)
        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=torch.float16):

                pipe.unet.half()
                noisy_images = noisy_images.half()
            
                predicted_noises = pipe.unet(noisy_images, t).sample  # Forward pass through the model
                noisy_images = pipe.scheduler.step(predicted_noises, t, noisy_images).prev_sample
                noisy_images = noisy_images.to(DEVICE)
    images = clamp(noisy_images)

    return images

def attack(image, superresolve_final=True):
    pil_image = Image.open(image)
    image = torchvision.transforms.functional.pil_to_tensor(pil_image)
    image = image.float() / 255.0
    image = image.to(DEVICE)
    image_expanded = image.unsqueeze(0)

    new_image = torch.zeros_like(image, device=DEVICE)
    for downsample_factor, alpha in blend:
        if downsample_factor == 1:
            new_image += image * alpha
            continue
        if alpha == 0:
            continue

        downsampled_image = torchvision.transforms.functional.resize(image_expanded, size//downsample_factor)
        for i in range(2):
            downsampled_image = diffusion_attack(downsampled_image, 100)
        downsampled_image = torchvision.transforms.functional.to_pil_image(downsampled_image[0])
        
        # super-resolution
        model = sr_models[downsample_factor]
        resampled_image = model.predict(downsampled_image.convert('RGB'))
        resampled_image = torchvision.transforms.functional.pil_to_tensor(resampled_image)
        resampled_image = resampled_image / 255
        resampled_image = resampled_image.to(DEVICE)

        new_image += resampled_image[0] * alpha

    new_image = torchvision.transforms.functional.to_pil_image(new_image)
    if superresolve_final:
        model = sr_models[2]
        new_image = model.predict(new_image)
        new_image = new_image.resize((512, 512))
    return new_image

def main(image_folder, output_folder, resume=True):
    images = os.listdir(image_folder)
    if not resume:
        if os.path.exists(output_folder):
            shutil.rmtree(output_folder)
        os.mkdir(output_folder)

    for image in tqdm(images):
        if resume and os.path.exists(os.path.join(output_folder, image)):
            continue
        if image.split(".")[-1] == "txt":
            continue
        attack(os.path.join(image_folder, image), superresolve_final=False).save(os.path.join(output_folder, image))

if __name__=="__main__":
    main("Neurips24_ETI_BeigeBox", "beigebox_results", resume=False)
