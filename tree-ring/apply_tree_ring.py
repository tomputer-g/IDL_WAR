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
):
    """Generates watermarked/non-watermarked images"""
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.mkdir(output_folder)
    os.mkdir(os.path.join(output_folder, "watermarked"))
    os.mkdir(os.path.join(output_folder, "unwatermarked"))
    os.mkdir(os.path.join(output_folder, "keys"))
    os.mkdir(os.path.join(output_folder, "masks"))

    from diffusers import (
        DPMSolverMultistepScheduler,
        DPMSolverMultistepInverseScheduler,
    )

    generator = TreeRingImageGenerator(
        scheduler=DPMSolverMultistepScheduler,
        inverse_scheduler=DPMSolverMultistepInverseScheduler,
    )
    captions, gt_ids = get_captions(
        hf_dataset,
        split=split,
        num_files_to_process=num_files_to_process,
        with_gt_id=True,
    )

    rng_generator = torch.cuda.manual_seed(0)
    unwatermarked_images, watermarked_images, keys, masks = [], [], [], []
    true_positive = 0
    false_positive = 0
    for i, caption in enumerate(captions):
        gt_id = os.path.basename(gt_ids[i])

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

    p_vals = []
    true_labels = []
    for i, watermarked, unwatermarked in zip(
        range(len(watermarked_images)), watermarked_images, unwatermarked_images
    ):
        unwatermarked_det = generator.detect(
            [unwatermarked], keys[i : i + 1], masks[i : i + 1], p_val_thresh=0.01
        )
        watermarked_det = generator.detect(
            [watermarked], keys[i : i + 1], masks[i : i + 1], p_val_thresh=0.01
        )

        p_vals.append(1 - unwatermarked_det[0][0])
        true_labels.append(0)

        false_positive += 1 if unwatermarked_det[0][1] else 0

        p_vals.append(1 - watermarked_det[0][0])
        true_labels.append(1)

        true_positive += 1 if watermarked_det[0][1] else 0

    print(f"AUC: {roc_auc_score(true_labels, p_vals)}")
    print(f"TPR: {true_positive / (true_positive + false_positive)}")


if __name__ == "__main__":
    main("phiyodr/coco2017", "", "outputs", split="validation", num_files_to_process=5)
