import os
import shutil
from datasets import load_dataset
from PIL import ImageFilter
from sklearn.metrics import roc_auc_score

from utils import visualize_tensor
from watermark import fft

import torch

# import click
from image_generators import TreeRingImageGenerator


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


# @click.command()
# @click.option("--input_file", help="File with prompts.")
# @click.option(
#     "--dataset",
#     default="ms-coco_train_2017",
#     help="Dataset you are working with, default ms-coco_train_2017",
# )
# @click.option("--output_folder", help="Folder to put watermarked images in.")
# @click.option("--num_files_to_process", help="The number of files to actually process.")
def main(
    hf_dataset,
    gt_folder,
    output_folder,
    split=None,
    num_files_to_process=-1,
    apply_watermark=True,
    resume=True
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

    from diffusers import (
        DPMSolverMultistepScheduler,
        DPMSolverMultistepInverseScheduler,
    )

    generator = TreeRingImageGenerator(
        scheduler=DPMSolverMultistepScheduler,
        inverse_scheduler=DPMSolverMultistepInverseScheduler,
        hyperparams={
            "half_precision": True,
        }
    )

    captions, gt_ids = get_captions(
        hf_dataset,
        split=split,
        num_files_to_process=num_files_to_process,
        with_gt_id=True,
    )

    rng_generator = torch.cuda.manual_seed(0)
    unwatermarked_images, watermarked_images, keys, masks = [], [], [], []
    for i, caption in enumerate(captions):
        gt_id = os.path.basename(gt_ids[i])

        if resume and os.path.exists(os.path.join(output_folder, "watermarked", gt_id)):
            continue

        image, key, mask = generator.generate_watermarked_images(
            [caption], rng_generator=rng_generator
        )
        watermarked_images.append(image[0])
        keys.append(key[0])
        masks.append(mask[0])

        image[0].save(os.path.join(output_folder, "watermarked", gt_id))
        torch.save(
            keys[i],
            os.path.join(output_folder, "keys", gt_id[:-4] + ".pt"),
        )
        torch.save(
            masks[i],
            os.path.join(output_folder, "masks", gt_id[:-4] + ".pt"),
        )

        image = generator.generate_images([caption], rng_generator=rng_generator)
        unwatermarked_images.append(image[0])

        image[0].save(os.path.join(output_folder, "unwatermarked", gt_id))

        with open(os.path.join(output_folder, "captions", gt_id[:-4] + ".txt"), mode="w") as f:
            f.write(caption)


if __name__ == "__main__":
    main("phiyodr/coco2017", "", "outputs", split="validation", num_files_to_process=-1, resume=True)
