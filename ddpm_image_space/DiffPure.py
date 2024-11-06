import torch
from guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults
from torchvision.utils import save_image
import torchvision.transforms as T
import os
import shutil
import yaml
import argparse
from PIL import Image
from tqdm import tqdm

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

class GuidedDiffusion(torch.nn.Module):
    def __init__(self, config, t, device=None, model_dir='pretrained/guided_diffusion'):
        super().__init__()
        # self.args = args
        self.config = config
        if device is None:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.device = device
        self.sample_step = 1
        self.t = t

        # load model
        model_config = model_and_diffusion_defaults()
        model_config.update(vars(self.config.model))
        # print(f'model_config: {model_config}')
        model, diffusion = create_model_and_diffusion(**model_config)
        model.load_state_dict(torch.load(f'{model_dir}/256x256_diffusion_uncond.pt', map_location='cpu'))
        model.requires_grad_(False).eval().to(self.device)

        if model_config['use_fp16']:
            model.convert_to_fp16()

        self.model = model
        self.diffusion = diffusion
        self.betas = torch.from_numpy(diffusion.betas).float().to(self.device)

    def image_editing_sample(self, img, bs_id=0, tag=None):
        with torch.no_grad():
            assert isinstance(img, torch.Tensor)
            batch_size = img.shape[0]

            assert img.ndim == 4, img.ndim
            img = img.to(self.device)
            x0 = img

            xs = []
            xts = []
            for it in range(self.sample_step):
                e = torch.randn_like(x0)
                total_noise_levels = self.t
                a = (1 - self.betas).cumprod(dim=0)
                x = x0 * a[total_noise_levels - 1].sqrt() + e * (1.0 - a[total_noise_levels - 1]).sqrt()

                xts.append(x.clone())

                for i in reversed(range(total_noise_levels)):
                    t = torch.tensor([i] * batch_size, device=self.device)

                    x = self.diffusion.p_sample(self.model, x, t,
                                                clip_denoised=True,
                                                denoised_fn=None,
                                                cond_fn=None,
                                                model_kwargs=None)["sample"]

                x0 = x
                xs.append(x0)

            return torch.cat(xs, dim=0), torch.cat(xts, dim=0)

class DiffPure():
    def __init__(self, steps=0.4, fname="base", output_dir=None):
        with open('configs/imagenet.yml', 'r') as f:
            config = yaml.safe_load(f)
        self.config = dict2namespace(config)
        self.runner = GuidedDiffusion(self.config, t = int(steps * int(self.config.model.timestep_respacing)), model_dir = 'pretrained')
        self.steps = steps
        self.save_imgs = output_dir is not None
        self.cnt = 0
        self.fname = fname
        self.output_dir = output_dir

        if self.save_imgs:
            if os.path.exists(output_dir+"_"+str(steps)):
                shutil.rmtree(output_dir+"_"+str(steps))
            os.mkdir(output_dir+"_"+str(steps))
            os.mkdir(os.path.join(output_dir+"_"+str(steps), "pured"))
            os.mkdir(os.path.join(output_dir+"_"+str(steps), "noisy"))
            os.mkdir(os.path.join(output_dir+"_"+str(steps), "original"))
                

    def __call__(self, img, name):
        img_pured, img_noisy = self.runner.image_editing_sample((img.unsqueeze(0) - 0.5) * 2)
        img_noisy = (img_noisy.squeeze(0).to(img.dtype).to("cpu") + 1) / 2
        img_pured = (img_pured.squeeze(0).to(img.dtype).to("cpu") + 1) / 2
        if self.save_imgs:
            save_dir = f'{self.output_dir}_{self.steps}'
            save_image(img, os.path.join(save_dir, "original", name))
            save_image(img_noisy, os.path.join(save_dir, "noisy", name))
            save_image(img_pured, os.path.join(save_dir, "pured", name))
            self.cnt += 1
        return img_pured
    
    def __repr__(self):
        return self.__class__.__name__ + '(steps={})'.format(self.steps)
    
def main(image_folder, output_folder, resume=True):
    images = os.listdir(image_folder)

    diff_attacker = DiffPure(steps=0.05, output_dir=output_folder)

    if not resume:
        if os.path.exists(output_folder):
            shutil.rmtree(output_folder)
        os.mkdir(output_folder)

    for image in tqdm(sorted(images)):
        if resume and os.path.exists(os.path.join(output_folder, image)):
            continue
        if image.split(".")[-1] == "txt":
            continue
        img = T.functional.pil_to_tensor(Image.open(os.path.join(image_folder, image)))
        img = img / 255.0
        diff_attacker(img, image)

if __name__=="__main__":
    main("../Neurips24_ETI_BeigeBox", "beigebox_results", resume=False)