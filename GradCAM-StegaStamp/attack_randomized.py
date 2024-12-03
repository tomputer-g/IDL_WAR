import glob
import numpy as np
import cv2
import os
from tqdm import tqdm

def attack(wm_img_filename, gradcam_filename, percentile_threshold, blur_size):
    wm_img = cv2.imread(wm_img_filename)

    attacked = wm_img.copy()
    gradcam = np.load(gradcam_filename)
    gradcam = np.abs(gradcam)

    if percentile_threshold < 0:
        mask = np.ones_like(attacked, dtype=bool)
    else:
        gradcam_mask = gradcam > np.percentile(gradcam, percentile_threshold)
        percentage = gradcam_mask.sum() / np.prod(gradcam_mask.shape)
        randomized = np.random.randn(*wm_img.shape)
        mask = randomized > percentage

    blurred_image = cv2.blur(attacked, (blur_size, blur_size))
    attacked[mask] = blurred_image[mask]
    
    # attacked = cv2.blur(attacked, (3, 3))

    return attacked    

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--wm_images_dir', type=str)
    parser.add_argument('--gradcams_dir', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--percentile_threshold', type=int, default=70)
    parser.add_argument('--blur_size', type=int, default=11)
    args = parser.parse_args()

    wm_files_list = sorted(glob.glob(args.wm_images_dir + '/*'), key=lambda x: os.path.basename(x).split(".")[0])

    output_dir = args.output_dir+f"_{args.percentile_threshold}_{args.blur_size}"
    os.makedirs(output_dir, exist_ok=True)

    for wm_filename in tqdm(wm_files_list):
        file_id = os.path.basename(wm_filename).split("_")[0]

        cv2.imwrite(
            os.path.join(output_dir, f"{file_id}_attacked.png"),
            attack(wm_filename, args.percentile_threshold, args.blur_size)
        )


if __name__ == "__main__":
    main()
