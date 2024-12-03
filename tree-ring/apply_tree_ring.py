import hashlib
import os
import shutil

import torch
from datasets import load_dataset
import click
from image_generators import get_tree_ring_generator
from PIL import ImageFilter
from utils import visualize_tensor
from watermark import fft

def get_unique_seed(i):
    i_str = str(i)
    hash_object = hashlib.sha256(i_str.encode())
    seed = int(hash_object.hexdigest(), 16) % (2**32-1)
    return seed

def get_captions(hf_dataset, split=None, num_files_to_process=-1, with_gt_id=False):
    ds = load_dataset(hf_dataset, split=split)

    if hf_dataset == "phiyodr/coco2017":
        if num_files_to_process != -1:
            to_process = ds[:num_files_to_process]
        else:
            to_process = ds

        captions = [caption_set[0] for caption_set in to_process["captions"]]
        gt_ids = to_process["file_name"]

        if with_gt_id:
            return captions, gt_ids
        else:
            return captions
    else:
        raise Exception("No caption getter implemented for the given dataset.")


@click.command()
@click.option(
    "--hf_dataset",
    default="phiyodr/coco2017",
    show_default=True,
    help="Dataset you are working with.",
)
@click.option("--split", default="validation", show_default=True, help="Split to use")
@click.option("--output_folder", default="outputs", show_default=True, help="Folder to put results in.")
@click.option("--channel", default=0, show_default=True, help="Channel to put tree-ring watermark in.")
@click.option("--num_files_to_process", default=-1, show_default=True, help="The number of files to actually process.")
@click.option("--resume", is_flag=True, show_default=True, default=True, help="Resume from previous run.")
@click.option("--model", default="stabilityai/stable-diffusion-2-1-base", show_default=True, help="Diffusion model to use")
@click.option("--scheduler", default="DPMSolverMultistepScheduler", show_default=True, help="Scheduler to use from [DPMSolverMultistepScheduler, DDIMScheduler]")
def main(
    hf_dataset,
    output_folder,
    split=None,
    num_files_to_process=-1,
    channel=0,
    resume=True,
    model="stabilityai/stable-diffusion-2-1-base",
    scheduler="DPMSolverMultistepScheduler",
):
    """Generates watermarked/non-watermarked images"""
    if not resume or not os.path.exists(output_folder):
        if os.path.exists(output_folder):
            shutil.rmtree(output_folder)
        os.mkdir(output_folder)
        os.mkdir(os.path.join(output_folder, "watermarked"))
        os.mkdir(os.path.join(output_folder, "unwatermarked"))
        os.mkdir(os.path.join(output_folder, "keys"))
        os.mkdir(os.path.join(output_folder, "masks"))
        os.mkdir(os.path.join(output_folder, "captions"))

    generator = get_tree_ring_generator(model, scheduler)

    captions, gt_ids = get_captions(
        hf_dataset,
        split=split,
        num_files_to_process=num_files_to_process,
        with_gt_id=True,
    )

    if os.path.exists(os.path.join(output_folder, "processed.txt")):
        with open(os.path.join(output_folder, "processed.txt"), mode="r") as f:
            processed = set([line.strip() for line in f.readlines()])
    else:
        processed = set()

    unwatermarked_images, watermarked_images, keys, masks = [], [], [], []
    for i, caption in enumerate(captions):
        gt_id = os.path.basename(gt_ids[i])

        if (resume and gt_id in processed):
            continue

        seed = get_unique_seed(i)
        rng_generator = torch.cuda.manual_seed(seed)
        image, key, mask = generator.generate_watermarked_images(
            [caption], rng_generator=rng_generator, channel=channel
        )
        watermarked_images.append(image[0])
        keys.append(key[0])
        masks.append(mask[0])

        image[0].save(os.path.join(output_folder, "watermarked", gt_id))
        torch.save(
            key[0],
            os.path.join(output_folder, "keys", gt_id[:-4] + ".pt"),
        )
        torch.save(
            mask[0],
            os.path.join(output_folder, "masks", gt_id[:-4] + ".pt"),
        )

        rng_generator = torch.cuda.manual_seed(seed)
        image = generator.generate_images([caption], rng_generator=rng_generator)
        unwatermarked_images.append(image[0])

        image[0].save(os.path.join(output_folder, "unwatermarked", gt_id))

        with open(os.path.join(output_folder, "captions", gt_id[:-4] + ".txt"), mode="w") as f:
            f.write(caption)
        
        with open(os.path.join(output_folder, "processed.txt"), mode="a") as f:
            f.write(gt_id+"\n")
        processed.add(gt_id)


if __name__ == "__main__":
    main()