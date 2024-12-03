import pickle
import multiprocessing
import os
import shutil

import click
from PIL import Image, UnidentifiedImageError

from image_generators import get_tree_ring_generator
from utils import visualize_latent

def save_latent(save_path, image_id, latents):
    # Save tensor to a .pkl file
    with open(os.path.join(save_path, f"{image_id}.pkl"), "wb") as f:
        pickle.dump(latents, f)

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

@click.command()
@click.option("--processed_file", default="outputs/processed.txt", show_default=True, help="Path to processed.txt")
@click.option("--unwatermarked_folder", default="outputs/unwatermarked", show_default=True, help="Path to unwatermarked images folder")
@click.option("--watermarked_folder", default="outputs/watermarked", show_default=True, help="Path to watermarked images folder")
@click.option("--output_folder", default="reversed_latents", show_default=True, help="Folder to output reversed latents to")
@click.option("--save_visualizations_instead", default=False, show_default=True, is_flag=True, help="Whether to only save the visualizations of the latents")
@click.option("--resume", is_flag=True, show_default=True, default=True, help="Resume from previous run.")
@click.option("--model", default="stabilityai/stable-diffusion-2-1-base", show_default=True, help="Diffusion model to use")
@click.option("--scheduler", default="DPMSolverMultistepScheduler", show_default=True, help="Scheduler to use from [DPMSolverMultistepScheduler, DDIMScheduler]")
def main(
    processed_file,
    unwatermarked_folder,
    watermarked_folder,
    output_folder,
    save_visualizations_instead=False,
    resume=True,
    model="stabilityai/stable-diffusion-2-1-base",
    scheduler="DPMSolverMultistepScheduler",
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

    generator = get_tree_ring_generator(model, scheduler)

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

        if not save_visualizations_instead:
            save_latent(
                os.path.join(output_folder, "unwatermarked"),
                image[:-4],
                unwatermarked_latents,
            )
            save_latent(
                os.path.join(output_folder, "watermarked"), image[:-4], watermarked_latents
            )
        else:
            visualize_latent(unwatermarked_latents, name=os.path.join(output_folder, "unwatermarked", image[:-4]))
            visualize_latent(watermarked_latents, name=os.path.join(output_folder, "watermarked", image[:-4]))

        processed.add(image)
        with open(os.path.join(output_folder, "processed.txt"), mode="a") as f:
            f.write(image + "\n")

    delete_temp_folder(unwatermarked_temp)
    delete_temp_folder(watermarked_temp)

if __name__ == "__main__":
    main()
