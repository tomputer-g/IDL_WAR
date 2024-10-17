import torch
from torchvision import transforms
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from diffusers import DDIMScheduler, DDIMInverseScheduler
from typing import Optional

def generate_initial_noise(
    batch_size: int,
    latent_shape: tuple[int, int, int] = (4, 64, 64),
    device: Optional[str] = None,
    dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    output_shape = (batch_size, *latent_shape)
    assert len(output_shape) == 4
    return torch.randn(output_shape, device=device, dtype=dtype)

def get_pipeline(
    device: Optional[str] = None,
    num_steps: int = 50,
    dtype: torch.dtype = torch.float16
) -> StableDiffusionPipeline:
    pipe = StableDiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2",
        torch_dtype=dtype,
    ).to(device)

    # pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.scheduler.set_timesteps(num_steps)
    pipe.scheduler.timesteps.to(device)

    return pipe
    

def denoise(
    pipe: StableDiffusionPipeline,
    prompts: list[str],
    latents: torch.Tensor,
    guidance_scale: float = 7.5,
    num_steps: int = 50,
):
    results = pipe(
        prompt=prompts,
        latents=latents,
        guidance_scale=guidance_scale,
        num_inference_steps=num_steps,
    )

    return results

def renoise(pipe, image_latents, guidance_scale=1.0, num_steps = 50):
    curr_scheduler = pipe.scheduler
    pipe.scheduler = DDIMInverseScheduler.from_config(pipe.scheduler.config)
    inverted_latents = pipe(
        prompt='',
        latents=image_latents,
        guidance_scale=guidance_scale,
        num_inference_steps=num_steps,
        output_type='latent',
    )

    pipe.scheduler = curr_scheduler
    return inverted_latents

# def renoise(pipe, image_latents, num_steps: int = 50):
    # print(image_latents)
    # inverted_latents = pipe.scheduler.add_noise(
    #     image_latents, 
    #     torch.randn_like(image_latents), 
    #     torch.tensor(num_steps-1)
    # )
    # for t in reversed(range(num_steps)):
    #     image_latents = pipe.scheduler.step(
    #         pipe.unet(image_latents, torch.tensor([t])).sample,
    #         torch.tensor([t]),
    #         image_latents
    #     ).prev_sample

    # return image_latents

def preprocess_images(images: list, device: Optional[str] = None):
    # Preprocess a batch of images
    preprocess = transforms.Compose([
        transforms.Resize((768, 768)),  # Resize all images to 512x512
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),  # Normalize to range [-1, 1]
    ])
    
    # Preprocess each image and stack them into a batch
    images = [preprocess(image) for image in images]
    image_tensor = torch.cat(images, dim=0).to(device)  # Create a batch by concatenating
    
    if len(image_tensor.size()) < 4:
        image_tensor = image_tensor.unsqueeze(0)

    return image_tensor

def get_latents(pipe, images):
    with torch.no_grad():
        latents = pipe.vae.encode(images).latent_dist.sample()
        latents = latents * 0.18215

    return latents

def get_images(pipe, latents):
    with torch.no_grad():
        images = pipe.vae.decode(latents / 0.18215).sample
        images = (images / 2 + 0.5).clamp(0, 1)
    return images


if __name__=='__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # prompts = ["Anime art of a dog in Shenandoah National Park"]
    # prompts = ["An astronaut riding a horse in Zion National Park"]
    prompts = ["White Pegasus eating from a large bowl of ice cream"]
    # prompts = ["A blue wailmer pokemon in the sea"]
    latents = generate_initial_noise(len(prompts), latent_shape=(4, 96, 96), device=device, dtype=torch.float32)
    pipe = get_pipeline(device, dtype=torch.float32)

    for i, img in enumerate(get_images(pipe, latents)):
        image = transforms.ToPILImage()(img.cpu())  # Convert tensor to PIL Image
        image.save(f"original_noise_{prompts[i]}.jpg", format="JPEG")

    
    imgs = denoise(pipe, prompts, latents)
    for i, img in enumerate(imgs.images):
        img.save(f"{prompts[i]}.jpg")

    # reverse
    images_tensor = preprocess_images(imgs.images, device=device)
    latents = get_latents(pipe, images_tensor)
    renoised = renoise(pipe, latents)
    # print(renoised.shape)

    for i, img in enumerate(get_images(pipe, renoised.images)):
        image = transforms.ToPILImage()(img.cpu())  # Convert tensor to PIL Image
        image.save(f"noise_{prompts[i]}.jpg", format="JPEG")
