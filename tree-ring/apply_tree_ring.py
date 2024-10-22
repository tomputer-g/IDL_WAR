import os
import shutil
from datasets import load_dataset
from PIL import ImageFilter

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

    generator = TreeRingImageGenerator()
    captions, gt_ids = get_captions(
        hf_dataset,
        split=split,
        num_files_to_process=num_files_to_process,
        with_gt_id=True,
    )

    rng_generator = torch.cuda.manual_seed(0)
    images, keys, masks = [], [], []
    detected = 0
    total = 0
    for caption in captions:
        image, key, mask = generator.generate_watermarked_images(
            [caption], rng_generator=rng_generator
        )
        images.append(image[0])
        keys.append(key[0])
        masks.append(mask[0])

        # image = generator.generate_images(
        #     [caption], generator=rng_generator
        # )
        # images.append(image[0])
        # keys.append(0)
        # masks.append(0)

        image[0] = image[0].filter(ImageFilter.GaussianBlur(radius=5))

        detection = generator.detect(image, key, mask, p_val_thresh=0.1)
        if detection[0][1]:
            detected += 1
        total += 1
    tpr = detected / total
    print(tpr)

    for img, key, mask, img_id in zip(images, keys, masks, gt_ids):
        img.save(os.path.join(output_folder, os.path.basename(img_id)))
        torch.save(
            key,
            os.path.join(output_folder, "key_" + os.path.basename(img_id)[:-4] + ".pt"),
        )
        torch.save(
            mask,
            os.path.join(output_folder, "mask_" + os.path.basename(img_id)[:-4])
            + ".pt",
        )


if __name__ == "__main__":
    main("phiyodr/coco2017", "", "outputs", split="validation", num_files_to_process=10)
