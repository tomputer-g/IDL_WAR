import pickle


def save_latent(save_path, image_id, latents):
    # Save tensor to a .pkl file
    with open(os.path.join(save_path, f"{image_id}.pkl"), "wb") as f:
        pickle.dump(latents, f)


import multiprocessing
import os
import shutil

import torch
from diffusers import DPMSolverMultistepInverseScheduler, DPMSolverMultistepScheduler
from PIL import Image, UnidentifiedImageError
from pytorch_fid.fid_score import calculate_fid_given_paths
from sklearn.metrics import auc, roc_curve

from image_generators import TreeRingImageGenerator


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
        print(
            f"{os.path.basename(img_path)} has a broken image file. Please regenerate."
        )


def copy_to_temp_folder_with_resize(
    images: list[str], original_folder: str, size=299
) -> str:
    temp_folder = "temp_resized_" + os.path.basename(original_folder)
    if os.path.exists(temp_folder):
        shutil.rmtree(temp_folder)
    os.mkdir(temp_folder)

    jobs = [
        (os.path.join(original_folder, image), size, os.path.join(temp_folder, image))
        for image in images
    ]

    with multiprocessing.Pool() as p:
        p.starmap(copy_resized, jobs)

    return temp_folder


def delete_temp_folder(temp_folder: str):
    shutil.rmtree(temp_folder)


def main(
    processed_file,
    unwatermarked_folder,
    watermarked_folder,
    output_folder,
    resume=True,
):
    if not resume or not os.path.exists(output_folder):
        if os.path.exists(output_folder):
            shutil.rmtree(output_folder)
        os.mkdir(output_folder)
        os.mkdir(os.path.join(output_folder, "unwatermarked"))
        os.mkdir(os.path.join(output_folder, "watermarked"))

    with open(processed_file, mode="r") as f:
        images = [line.strip() for line in f.readlines()]

    if os.path.exists(os.path.join(output_folder, "processed.txt")):
        with open(os.path.join(output_folder, "processed.txt"), mode="r") as f:
            processed = set([line.strip() for line in f.readlines()])
    else:
        processed = set()

    unwatermarked_temp = copy_to_temp_folder(images, unwatermarked_folder)
    watermarked_temp = copy_to_temp_folder(images, watermarked_folder)

    generator = TreeRingImageGenerator(
        scheduler=DPMSolverMultistepScheduler,
        inverse_scheduler=DPMSolverMultistepInverseScheduler,
        hyperparams={
            "half_precision": True,
        },
    )

    # load masks + prune broken masks/images
    for image in images[:]:
        if image in processed:
            continue

        try:
            unwatermarked = Image.open(os.path.join(unwatermarked_temp, image))
            watermarked = Image.open(os.path.join(watermarked_temp, image))
        except UnidentifiedImageError:
            print(f"{image} has a broken image file. Please regenerate.")
            images.remove(image)
            continue

        unwatermarked_latents = generator.renoise_images([unwatermarked])
        watermarked_latents = generator.renoise_images([watermarked])

        save_latent(
            os.path.join(output_folder, "unwatermarked"),
            image[:-4],
            unwatermarked_latents,
        )
        save_latent(
            os.path.join(output_folder, "watermarked"), image[:-4], watermarked_latents
        )

        processed.add(image)
        with open(os.path.join(output_folder, "processed.txt"), mode="a") as f:
            f.write(image + "\n")

    delete_temp_folder(unwatermarked_temp)
    delete_temp_folder(watermarked_temp)


if __name__ == "__main__":
    main(
        "outputs_original/processed.txt",
        "outputs_original/unwatermarked",
        "outputs_original/watermarked",
        "reversed_latents",
    )