import torch
from torchvision import transforms
from inversable_stable_diffusion import InversableStableDiffusionPipeline
from diffusers import (DPMSolverMultistepInverseScheduler,
                           DPMSolverMultistepScheduler)
from diffusers import (DDIMScheduler, DDIMInverseScheduler)
from typing import Optional
from PIL.Image import Image
from functools import partial


from watermark import watermark, detect_pval, detect_dist, extract_key


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

    def generate_images(
        self, prompts: list[str], rng_generator: torch.Generator
    ) -> list[Image]:
        latents = self._generate_initial_noise(len(prompts), rng_generator)

        # for i, img in enumerate(self._get_images(latents)):
        #     image = transforms.ToPILImage()(img.cpu())  # Convert tensor to PIL Image
        #     image.save(f"original_noise_{prompts[i]}.jpg", format="JPEG")

        imgs = self._denoise(prompts, latents)

        # for i, img in enumerate(imgs.images):
        #     img.save(f"image_{prompts[i]}.jpg")

        self.prompts = prompts  # used for saving debugging images later

        del latents
        return imgs.images

    def renoise_images(self, images: list[Image]) -> list[torch.Tensor]:
        images_tensor = self._preprocess_images_for_renoising(images)
        latents = self._get_latents(images_tensor)
        renoised = self._renoise(latents)

        # for i, img in enumerate(self._get_images(renoised.images)):
        #     image = transforms.ToPILImage()(img.cpu())  # Convert tensor to PIL Image
        #     image.save(f"noise_{self.prompts[i]}.jpg", format="JPEG")

        del latents
        del images_tensor
        return renoised

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

    def _get_pipeline(self, model, scheduler) -> InversableStableDiffusionPipeline:
        pipe = InversableStableDiffusionPipeline.from_pretrained(
            model,
            torch_dtype=self.dtype,
        ).to(self.device)

        pipe.scheduler = scheduler.from_pretrained(model, subfolder="scheduler")

        # pipe.scheduler = scheduler.from_config(pipe.scheduler.config)
        # pipe.scheduler.set_timesteps(self.num_steps)
        # pipe.scheduler.timesteps.to(self.device)

        return pipe

    def _generate_initial_noise(
        self, batch_size: int, generator: torch.Generator
    ) -> torch.Tensor:
        return self.pipe.prepare_latents(
            batch_size,
            self.latent_shape[0],
            self.resolution,
            self.resolution,
            self.dtype,
            self.device,
            generator
        )

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
        with torch.no_grad():
            text_embeddings = self.pipe.get_text_embedding("")
            inverted_latents = self.pipe.forward_diffusion(
                text_embeddings=text_embeddings,
                latents=image_latents,
                guidance_scale=self.renoise_guidance_scale,
                num_inference_steps=self.num_steps,
            )

        return inverted_latents

    def _preprocess_images_for_renoising(self, images: list[Image]) -> torch.Tensor:
        # Preprocess a batch of images
        preprocess = transforms.Compose(
            [
                transforms.Resize(self.resolution),  # Resize all images to 512x512
                transforms.ToTensor(),
            ]
        )

        # Preprocess each image and stack them into a batch
        images = [preprocess(image) for image in images]
        image_tensor = torch.stack(images).to(
            self.device
        )  # Create a batch by concatenating
        image_tensor = image_tensor.type(self.dtype)

        if len(image_tensor.size()) < 4:
            image_tensor = image_tensor.unsqueeze(0)

        return image_tensor

    def _get_latents(self, images: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            latents = self.pipe.get_image_latents(images, sample=False)

        return latents

    def _get_images(self, latents: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            images = self.pipe.vae.decode(latents / 0.18215).mode
            images = (images / 2 + 0.5).clamp(0, 1)
        return images


class TreeRingImageGenerator(ImageGenerator):
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
        tree_ring_hyperparams: dict[str, any] = {
            "type": "rings",
            "radius": 10,
        },
    ):
        super().__init__(
            model=model,
            scheduler=scheduler,
            inverse_scheduler=inverse_scheduler,
            hyperparams=hyperparams,
        )

        self.type = tree_ring_hyperparams.get("type", "rings")
        self.radius = tree_ring_hyperparams.get("radius", 10)

    def generate_watermarked_images(
        self,
        prompts: list[str],
        rng_generator: Optional[torch.Generator] = None,
        channel: int = 0,
        key: torch.Tensor = None,
        visualize_key: bool = False,
        visualization_folder: str = "",
        visualization_name: str = "",
    ) -> tuple[list[Image], list[torch.Tensor], list[torch.Tensor]]:
        latents = self._generate_initial_noise(len(prompts), rng_generator)

        keys = []
        masks = []
        for i in range(latents.shape[0]):
            tensor = latents[i]
            assert tensor.shape == self.latent_shape

            tensor, key, mask = watermark(
                tensor,
                self.type,
                self.radius,
                device=self.device,
                channel=channel,
                key=key,
                visualize_key=visualize_key,
                visualization_folder=visualization_folder,
                visualization_name=visualization_name,
            )
            latents[i] = tensor
            keys.append(key)
            masks.append(mask)

        imgs = self._denoise(prompts, latents)

        # for i, img in enumerate(imgs.images):
        #     img.save(f"watermarked_image_{prompts[i]}.jpg")

        self.prompts = prompts  # used for saving debugging images later

        del latents
        return imgs.images, keys, masks

    def detect(
        self,
        images: list[Image],
        keys: list[torch.Tensor],
        masks: list[torch.Tensor],
        use_pval: bool = True,
        p_val_thresh: float = 0.01,
        dist_thresh: float = 77,
        visualize_renoised: bool = False,
        visualization_folder: str = "",
        visualization_name: str = "",
    ) -> list[bool]:
        if use_pval:
            detect = partial(
                detect_pval,
                p_val_thresh=p_val_thresh,
                visualize_renoised=visualize_renoised,
                visualization_folder=visualization_folder,
                visualization_name=visualization_name
            )
        else:
            detect = partial(
                detect_dist,
                dist_thresh=dist_thresh,
                visualize_renoised=visualize_renoised,
                visualization_folder=visualization_folder,
                visualization_name=visualization_name
            )

        latents = self.renoise_images(images)

        results = []
        for i in range(len(images)):
            results.append(
                detect(latents[i], keys[i], masks[i])
            )
        return results
    
    def extract_key(self, images: list[Image], masks: list[torch.Tensor]) -> list[torch.Tensor]:
        latents = self.renoise_images(images)

        results = []
        for i in range(len(images)):
            results.append(
                extract_key(latents[i], masks[i])
            )
        return results

def get_tree_ring_generator(model, scheduler):
    # "stabilityai/stable-diffusion-2-1-base"

    if scheduler == "DPMSolverMultistepScheduler":
        scheduler = DPMSolverMultistepScheduler
        inverse_scheduler = DPMSolverMultistepInverseScheduler
    elif scheduler == "DDIMScheduler":
        scheduler = DDIMScheduler
        inverse_scheduler = DDIMInverseScheduler
    else:
        raise Exception(f"Not supported scheduler {scheduler} requested")

    generator = TreeRingImageGenerator(
        model=model,
        scheduler=scheduler,
        inverse_scheduler=inverse_scheduler,
        hyperparams={
            "half_precision": True,
        }
    )

    return generator

if __name__ == "__main__":
    # from diffusers import DPMSolverMultistepScheduler, DPMSolverMultistepInverseScheduler
    device = "cuda" if torch.cuda.is_available() else "cpu"

    prompts = [
        "Anime art of a dog in Shenandoah National Park",
        "An astronaut riding a horse in Zion National Park",
        "White Pegasus eating from a large bowl of ice cream",
        "A blue wailmer pokemon in the sea",
    ]

    generator = TreeRingImageGenerator(
        model="stabilityai/stable-diffusion-2",
        hyperparams={"resolution": 512, "denoise_guidance_scale": 7.5},
    )

    for prompt in prompts:
        rng_generator = torch.cuda.manual_seed(123)
        images = generator.generate_images([prompt], rng_generator)

        watermarked_images, keys, masks = generator.generate_watermarked_images(
            [prompt], seed=123
        )

        print(generator.detect(images, keys, masks))
        print(generator.detect(watermarked_images, keys, masks))
