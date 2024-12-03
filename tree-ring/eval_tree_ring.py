import multiprocessing
import os
import shutil

import numpy as np
import torch
import click
from diffusers import (DPMSolverMultistepInverseScheduler,
                       DPMSolverMultistepScheduler)
from diffusers import (DDIMScheduler, DDIMInverseScheduler)
from PIL import Image, UnidentifiedImageError
from pytorch_fid.fid_score import calculate_fid_given_paths
from sklearn.metrics import auc, roc_curve

from image_generators import get_tree_ring_generator

def get_attack(attack_name):
    if attack_name == "none":
        return lambda img: img
    if attack_name == "rotation":
        return lambda img: img.rotate(75)
    if attack_name == "blur":
        return lambda img: img.filter(ImageFilter.GaussianBlur(radius=4))
    
    raise Exception(f"Unimplemented attack {attack_name} requested")

def copy_to_temp_folder(images: list[str], original_folder: str) -> str:
    temp_folder = "temp_" + os.path.basename(original_folder)
    if os.path.exists(temp_folder):
        shutil.rmtree(temp_folder)
    os.mkdir(temp_folder)

    for image in images:
        shutil.copy2(
            os.path.join(original_folder, image), os.path.join(temp_folder, image)
        )
    return temp_folder


def copy_resized(img_path, size, img_dest, attack=None):
    try:
        with Image.open(img_path) as img:
            if attack is not None:
                img = attack(img)
            resized_img = img.resize((size, size))
            resized_img.save(img_dest)
    except UnidentifiedImageError:
        print(
            f"{os.path.basename(img_path)} has a broken image file. Please regenerate."
        )


def copy_to_temp_folder_with_resize(
    images: list[str], original_folder: str, size=299, attack=None
) -> str:
    temp_folder = "temp_resized_" + os.path.basename(original_folder)
    if os.path.exists(temp_folder):
        shutil.rmtree(temp_folder)
    os.mkdir(temp_folder)

    jobs = [
        (os.path.join(original_folder, image), size, os.path.join(temp_folder, image), attack)
        for image in images
    ]

    with multiprocessing.Pool() as p:
        p.starmap(copy_resized, jobs)

    return temp_folder


def delete_temp_folder(temp_folder: str):
    shutil.rmtree(temp_folder)


def eval_auc_and_tpr(
    images,
    unwatermarked_folder,
    watermarked_folder,
    keys,
    masks,
    fpr_target=0.01,
    precalculated_data_files={
        "unwatermarked": "eval_probs_unwatermarked.csv",
        "watermarked": "eval_probs_watermarked.csv"    
    },
    new_data_files={
        "unwatermarked": "eval_probs_unwatermarked.csv",
        "watermarked": "eval_probs_watermarked.csv",
        "combined": "eval_probs.csv",  
    },
    attack=None,
    model="stabilityai/stable-diffusion-2-1-base",
    scheduler="DPMSolverMultistepScheduler",
):
    generator = get_tree_ring_generator(model, scheduler)

    # generator = TreeRingImageGenerator(
    #     model="stabilityai/stable-diffusion-2-1-base",
    #     scheduler=DDIMScheduler,
    #     inverse_scheduler=DDIMInverseScheduler,
    #     hyperparams={
    #         "half_precision": True,
    #     }
    # )

    probabilities = []
    true_labels = []

    precalculated_data = {}
    for precalculated_data_file in precalculated_data_files.values():
        if not os.path.exists(precalculated_data_file):
            continue

        with open(precalculated_data_file, mode="r") as f:
            precalculated_data.update({
                (
                    line.split(",")[0],  # image label
                    line.split(",")[1],  # "watermarked" or "unwatermarked"
                ): (
                    int(line.split(",")[2]),  # true_label
                    float(line.split(",")[3].strip()),  # p_val
                )
                for line in f
            })

    for true_label, p_val in precalculated_data.values():
        # probabilities.append(1 - p_val)
        watermarked_prob = -p_val
        true_labels.append(true_label)

    for i, image in enumerate(images):
        try:
            unwatermarked = Image.open(os.path.join(unwatermarked_folder, image))
            watermarked = Image.open(os.path.join(watermarked_folder, image))

            if attack is not None:
                watermarked.save("debug_unattacked.jpg")
                unwatermarked = attack(unwatermarked)
                watermarked = attack(watermarked)
                watermarked.save("debug_attacked.jpg")

        except UnidentifiedImageError:
            print(f"{image} has a broken image file. Please regenerate.")
            continue

        if (image, "unwatermarked") not in precalculated_data:
            p_val, _ = generator.detect(
                [unwatermarked], keys[i : i + 1], masks[i : i + 1], use_pval=True
            )[0]
            watermarked_prob = 1 - p_val
            # watermarked_prob = -p_val

            true_labels.append(0)
            probabilities.append(watermarked_prob)

            with open(new_data_files["unwatermarked"], mode="a") as f:
                f.write(f"{image},unwatermarked,{0},{p_val}\n")
            with open(new_data_files["combined"], mode="a") as f:
                f.write(f"{image},unwatermarked,{0},{p_val}\n")

        if (image, "watermarked") not in precalculated_data:
            p_val, _ = generator.detect(
                [watermarked], keys[i : i + 1], masks[i : i + 1], use_pval=True
            )[0]
            watermarked_prob = 1 - p_val
            # watermarked_prob = -p_val

            true_labels.append(1)
            probabilities.append(watermarked_prob)

            with open(new_data_files["watermarked"], mode="a") as f:
                f.write(f"{image},watermarked,{1},{p_val}\n")
            with open(new_data_files["combined"], mode="a") as f:
                f.write(f"{image},watermarked,{1},{p_val}\n")

    fpr, tpr, _ = roc_curve(true_labels, probabilities)
    auc_val = auc(fpr, tpr)
    tpr_at_fpr = tpr[np.where(fpr<fpr_target)[0][-1]]

    return auc_val, tpr_at_fpr


def eval_fid(gt_folder, unwatermarked_folder, watermarked_folder):
    unwatermarked_fid = calculate_fid_given_paths(
        [gt_folder, unwatermarked_folder],
        50,
        "cuda" if torch.cuda.is_available() else "cpu",
        2048,
    )

    watermarked_fid = calculate_fid_given_paths(
        [gt_folder, watermarked_folder],
        50,
        "cuda" if torch.cuda.is_available() else "cpu",
        2048,
    )

    return unwatermarked_fid, watermarked_fid

@click.command()
@click.option("--processed_file", default="outputs/processed.txt", show_default=True, help="Path to processed.txt")
@click.option("--gt_folder", default="val2017", show_default=True, help="Path to ground truth folder")
@click.option("--unwatermarked_folder", default="outputs/unwatermarked", show_default=True, help="Path to unwatermarked images folder")
@click.option("--watermarked_folder", default="outputs/watermarked", show_default=True, help="Path to watermarked images folder")
@click.option("--keys_folder", default="outputs/keys", show_default=True, help="Path to keys folder")
@click.option("--masks_folder", default="outputs/masks", show_default=True, help="Path to masks folder")
@click.option("--attack", default="none", show_default=True, help="Attack to evaluate against from [none, rotation, blur]")
@click.option("--model", default="stabilityai/stable-diffusion-2-1-base", show_default=True, help="Diffusion model to use")
@click.option("--scheduler", default="DPMSolverMultistepScheduler", show_default=True, help="Scheduler to use from [DPMSolverMultistepScheduler, DDIMScheduler]")
def main(
    processed_file,
    gt_folder,
    unwatermarked_folder,
    watermarked_folder,
    keys_folder,
    masks_folder,
    attack=None,
    model="stabilityai/stable-diffusion-2-1-base",
    scheduler="DPMSolverMultistepScheduler",
):
    
    if attack is not None:
        attack = get_attack(attack)

    with open(processed_file, mode="r") as f:
        images = [line.strip() for line in f.readlines()]

    gt_temp = copy_to_temp_folder(images, gt_folder)
    unwatermarked_temp = copy_to_temp_folder(images, unwatermarked_folder)
    watermarked_temp = copy_to_temp_folder(images, watermarked_folder)
    keys_temp = copy_to_temp_folder(
        [image[:-4] + ".pt" for image in images], keys_folder
    )
    masks_temp = copy_to_temp_folder(
        [image[:-4] + ".pt" for image in images], masks_folder
    )

    # load keys/masks + prune broken keys/masks/images
    keys = []
    masks = []
    for image in images[:]:
        try:
            key = torch.load(
                os.path.join(keys_temp, image[:-4] + ".pt"), weights_only=True
            )
        except EOFError:
            print(f"{image} has a broken key file. Please regenerate.")
            images.remove(image)
            continue

        try:
            mask = torch.load(
                os.path.join(masks_temp, image[:-4] + ".pt"), weights_only=True
            )
        except EOFError:
            print(f"{image} has a broken key file. Please regenerate.")
            images.remove(image)
            continue

        try:
            Image.open(os.path.join(unwatermarked_temp, image))
            Image.open(os.path.join(watermarked_temp, image))
        except UnidentifiedImageError:
            print(f"{image} has a broken image file. Please regenerate.")
            images.remove(image)
            continue

        keys.append(key)
        masks.append(mask)

    auc_val, tpr = eval_auc_and_tpr(
        images,
        unwatermarked_temp,
        watermarked_temp,
        keys,
        masks,
        attack=attack,
        model=model,
        scheduler=scheduler,
    )

    delete_temp_folder(gt_temp)
    delete_temp_folder(unwatermarked_temp)
    delete_temp_folder(watermarked_temp)
    delete_temp_folder(keys_temp)
    delete_temp_folder(masks_temp)

    gt_temp_resized = copy_to_temp_folder_with_resize(images, gt_folder)
    unwatermarked_temp_resized = copy_to_temp_folder_with_resize(
        images, unwatermarked_folder, attack=attack
    )
    watermarked_temp_resized = copy_to_temp_folder_with_resize(
        images, watermarked_folder, attack=attack
    )

    unwatermarked_fid, watermarked_fid = eval_fid(
        gt_temp_resized, unwatermarked_temp_resized, watermarked_temp_resized
    )

    delete_temp_folder(gt_temp_resized)
    delete_temp_folder(unwatermarked_temp_resized)
    delete_temp_folder(watermarked_temp_resized)

    print(f"AUC: {auc_val}")
    print(f"TPR@1%FPR: {tpr}")
    print(f"Unwatermarked FID: {unwatermarked_fid}")
    print(f"Watermarked FID: {watermarked_fid}")


if __name__ == "__main__":
    main()
