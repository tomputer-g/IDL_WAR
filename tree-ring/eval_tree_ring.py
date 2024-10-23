import multiprocessing
import os
import shutil

import torch
from diffusers import (DPMSolverMultistepInverseScheduler,
                       DPMSolverMultistepScheduler)
from image_generators import TreeRingImageGenerator
from PIL import Image, UnidentifiedImageError
from pytorch_fid.fid_score import calculate_fid_given_paths
from sklearn.metrics import auc, roc_curve


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

def copy_resized(img_path, size, img_dest):
    try:
        with Image.open(img_path) as img:
            resized_img = img.resize((size, size))
            resized_img.save(img_dest)
    except UnidentifiedImageError:
        print(f"{os.path.basename(img_path)} has a broken image file. Please regenerate.")

def copy_to_temp_folder_with_resize(
    images: list[str], original_folder: str, size=299
) -> str:
    temp_folder = "temp_resized_" + os.path.basename(original_folder)
    if os.path.exists(temp_folder):
        shutil.rmtree(temp_folder)
    os.mkdir(temp_folder)

    jobs = [(
        os.path.join(original_folder, image),
        size,
        os.path.join(temp_folder, image)
    ) for image in images]
    
    with multiprocessing.Pool() as p:
        p.starmap(copy_resized, jobs)

    return temp_folder


def delete_temp_folder(temp_folder: str):
    shutil.rmtree(temp_folder)


def eval_auc_and_tpr(
    images, unwatermarked_folder, watermarked_folder, keys, masks, fpr_target=0.01
):
    generator = TreeRingImageGenerator(
        scheduler=DPMSolverMultistepScheduler,
        inverse_scheduler=DPMSolverMultistepInverseScheduler,
        hyperparams={
            "half_precision": True,
        },
    )

    probabilities = []
    true_labels = []
    for i, image in enumerate(images[:2]):
        try:
            unwatermarked = Image.open(os.path.join(unwatermarked_folder, image))
            watermarked = Image.open(os.path.join(watermarked_folder, image))
        except UnidentifiedImageError:
            print(f"{image} has a broken key file. Please regenerate.")
            continue
        
        p_val, _ = generator.detect(
            [unwatermarked], keys[i : i + 1], masks[i : i + 1], p_val_thresh=0.01
        )[0]

        true_labels.append(0)
        probabilities.append(1 - p_val)

        p_val, _ = generator.detect(
            [watermarked], keys[i : i + 1], masks[i : i + 1], p_val_thresh=0.01
        )[0]

        true_labels.append(1)
        probabilities.append(1 - p_val)

    fpr, tpr, _ = roc_curve(true_labels, probabilities)
    auc_val = auc(fpr, tpr)
    tpr_at_fpr = tpr[next(i for i, x in enumerate(fpr) if x >= fpr_target)]

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


def main(
    processed_file,
    gt_folder,
    unwatermarked_folder,
    watermarked_folder,
    keys_folder,
    masks_folder,
):
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
            key = torch.load(os.path.join(keys_temp, image[:-4] + ".pt"), weights_only=True)
        except EOFError:
            print(f"{image} has a broken key file. Please regenerate.")
            images.remove(image)
            continue

        try:
            mask = torch.load(os.path.join(masks_temp, image[:-4] + ".pt"), weights_only=True)
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
        images, unwatermarked_temp, watermarked_temp, keys, masks
    )

    delete_temp_folder(gt_temp)
    delete_temp_folder(unwatermarked_temp)
    delete_temp_folder(watermarked_temp)
    delete_temp_folder(keys_temp)
    delete_temp_folder(masks_temp)

    gt_temp_resized = copy_to_temp_folder_with_resize(images, gt_folder)
    unwatermarked_temp_resized = copy_to_temp_folder_with_resize(
        images, unwatermarked_folder
    )
    watermarked_temp_resized = copy_to_temp_folder_with_resize(
        images, watermarked_folder
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
    main(
        "outputs/processed.txt",
        "val2017",
        "outputs/unwatermarked",
        "outputs/watermarked",
        "outputs/keys",
        "outputs/masks",
    )
