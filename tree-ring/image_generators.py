import torch
from torchvision import transforms
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from diffusers import DDIMScheduler, DDIMInverseScheduler
from typing import Optional, Iterable
from PIL.Image import Image


class ImageGenerator:
    def __init__(
        self,
        model: str = "stabilityai/stable-diffusion-2",
        scheduler=DDIMScheduler,
        inverse_scheduler=DDIMInverseScheduler,
        hyperparams: dict[str, any] = {
            "resolution": 512,
            "num_steps": 50,
            "device": None,
            "half_precision": False,
            "denoise_guidance_scale": 7.5,
            "renoise_guidance_scale": 1.0,
        },
    ):
        self._set_hyperparams(**hyperparams)

        self.pipe = self._get_pipeline(model, scheduler)
        self.model = model
        self.scheduler = scheduler
        self.inverse_scheduler = inverse_scheduler

    def generate_images(self, prompts: list[str]) -> list[Image]:
        latents = self._generate_initial_noise(len(prompts))

        for i, img in enumerate(self._get_images(latents)):
            image = transforms.ToPILImage()(img.cpu())  # Convert tensor to PIL Image
            image.save(f"original_noise_{prompts[i]}.jpg", format="JPEG")

        imgs = self._denoise(prompts, latents)
        

        for i, img in enumerate(imgs.images):
            img.save(f"image_{prompts[i]}.jpg")

        self.prompts = prompts # used for saving debugging images later

        del latents
        return imgs.images

    def renoise_images(self, images: list[Image]) -> list[torch.Tensor]:
        images_tensor = self._preprocess_images_for_renoising(images)
        latents = self._get_latents(images_tensor)
        renoised = self._renoise(latents)

        for i, img in enumerate(self._get_images(renoised.images)):
            image = transforms.ToPILImage()(img.cpu())  # Convert tensor to PIL Image
            image.save(f"noise_{self.prompts[i]}.jpg", format="JPEG")

        del latents
        del images_tensor
        return renoised.images

    def _set_hyperparams(
        self,
        resolution: int = 512,
        num_steps: int = 50,
        device: Optional[str] = None,
        half_precision: bool = False,
        denoise_guidance_scale: float = 7.5,
        renoise_guidance_scale: float = 1.0,
    ):
        # resolution settings
        self.resolution = resolution

        # latent tensor size based on resolution
        match resolution:
            case 512:
                self.latent_shape = (4, 64, 64)
            case 768:
                self.latent_shape = (4, 96, 96)
            case _:
                raise Exception(
                    f"Invalid resolution of {resolution}. Must be 512 or 768"
                )

        # number of noising/denoising steps
        self.num_steps = num_steps

        # cuda vs cpu settings
        self.device = device
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # half precision settings
        self.half_precision = half_precision
        if half_precision:
            self.dtype = torch.float16
        else:
            self.dtype = torch.float32

        # guidance scale settings
        self.denoise_guidance_scale = denoise_guidance_scale
        self.renoise_guidance_scale = renoise_guidance_scale

    def _get_pipeline(self, model, scheduler) -> StableDiffusionPipeline:
        pipe = StableDiffusionPipeline.from_pretrained(
            model,
            torch_dtype=self.dtype,
        ).to(self.device)

        pipe.scheduler = scheduler.from_config(pipe.scheduler.config)
        pipe.scheduler.set_timesteps(self.num_steps)
        pipe.scheduler.timesteps.to(self.device)

        return pipe

    def _generate_initial_noise(self, batch_size: int) -> torch.Tensor:
        output_shape = (batch_size, *self.latent_shape)
        assert len(output_shape) == 4
        return torch.randn(output_shape, device=self.device, dtype=self.dtype)

    def _denoise(self, prompts: list[str], latents: torch.Tensor) -> list:
        with torch.no_grad():
            results = self.pipe(
                prompt=prompts,
                latents=latents,
                guidance_scale=self.denoise_guidance_scale,
                num_inference_steps=self.num_steps,
                height=self.resolution,
                width=self.resolution,
            )

        return results

    def _renoise(self, image_latents) -> list:
        curr_scheduler = self.pipe.scheduler
        self.pipe.scheduler = self.inverse_scheduler.from_config(self.pipe.scheduler.config)

        with torch.no_grad():
            inverted_latents = self.pipe(
                prompt="",
                latents=image_latents,
                guidance_scale=self.renoise_guidance_scale,
                num_inference_steps=self.num_steps,
                output_type="latent",
            )

        self.pipe.scheduler = curr_scheduler
        return inverted_latents

    def _preprocess_images_for_renoising(self, images: list[Image]) -> torch.Tensor:
        # Preprocess a batch of images
        preprocess = transforms.Compose(
            [
                transforms.Resize(self.resolution),  # Resize all images to 512x512
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),  # Normalize to range [-1, 1]
            ]
        )

        # Preprocess each image and stack them into a batch
        images = [preprocess(image) for image in images]
        image_tensor = torch.stack(images).to(
            self.device
        )  # Create a batch by concatenating

        if len(image_tensor.size()) < 4:
            image_tensor = image_tensor.unsqueeze(0)

        return image_tensor

    def _get_latents(self, images: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            latents = self.pipe.vae.encode(images).latent_dist.sample()
            latents = latents * 0.18215

        return latents

    def _get_images(self, latents: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            images = self.pipe.vae.decode(latents / 0.18215).sample
            images = (images / 2 + 0.5).clamp(0, 1)
        return images


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    prompts = [
        "Anime art of a dog in Shenandoah National Park",
        "An astronaut riding a horse in Zion National Park",
        "White Pegasus eating from a large bowl of ice cream",
        "A blue wailmer pokemon in the sea",
    ]

    generator = ImageGenerator(hyperparams={"resolution": 768})

    for prompt in prompts:
        images = generator.generate_images([prompt])
        generator.renoise_images(images)
