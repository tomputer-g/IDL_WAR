import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
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

def get_pipeline(device: Optional[str] = None, dtype: torch.dtype = torch.float16) -> StableDiffusionPipeline:
    pipe = StableDiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2",
        torch_dtype=dtype,
    ).to(device)

    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)


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
        # height=512,
        # width=512,
    )

    return results

if __name__=='__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # prompts = ["Anime art of a dog in Shenandoah National Park"]
    # prompts = ["An astronaut riding a horse in Zion National Park"]
    prompts = ["White Pegasus eating from a large bowl of ice cream"]
    # prompts = ["A blue wailmer pokemon in the sea"]
    latents = generate_initial_noise(len(prompts), latent_shape=(4, 96, 96), device=device, dtype=torch.float32)
    pipe = get_pipeline(device, dtype=torch.float32)
    # pipe.enable_attention_slicing()
    imgs = denoise(pipe, prompts, latents)
    # imgs = pipe(
    #     prompt=prompts,
    #     # latents=latents,
    #     guidance_scale=7.5,
    #     num_inference_steps=50,
    #     # height=512,
    #     # width=512,
    # )
    for i, img in enumerate(imgs.images):
        img.save(f"{prompts[i]}.jpg")
    # imgs.images[0].save("cat.jpg")