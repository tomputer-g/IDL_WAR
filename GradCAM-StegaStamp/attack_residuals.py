import glob
import numpy as np
import cv2
import os

def attack(wm_img, unwm_img, percentile_threshold, blur_size, for_residual=None):
    if for_residual is None:
        for_residual = wm_img

    attacked = wm_img.copy()
    residual = cv2.absdiff(for_residual, unwm_img)
    if percentile_threshold < 0:
        mask = np.ones_like(attacked, dtype=bool)
    else:
        mask = residual > np.percentile(residual, percentile_threshold)

    blurred_image = cv2.blur(attacked, (blur_size, blur_size))
    attacked[mask] = blurred_image[mask]
    
    # attacked = cv2.blur(attacked, (3, 3))

    return attacked    

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--wm_images_dir', type=str)
    parser.add_argument('--wm_images_for_residuals_dir', type=str, default=None)
    parser.add_argument('--unwm_images_dir', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--percentile_threshold', type=int, default=70)
    parser.add_argument('--blur_size', type=int, default=11)
    args = parser.parse_args()

    if args.wm_images_for_residuals_dir is None:
        args.wm_images_for_residuals_dir = args.wm_images

    # wm_files_list = sorted(glob.glob(args.wm_images_dir + '/*'), key=lambda x: os.path.basename(x).split(".")[0])
    unwm_files_list = sorted(glob.glob(args.unwm_images_dir + '/*'), key=lambda x: os.path.basename(x).split(".")[0])

    output_dir = args.output_dir+f"_{args.percentile_threshold}_{args.blur_size}"
    os.makedirs(output_dir, exist_ok=True)

    for unwm_filename in unwm_files_list:
        file_id = os.path.basename(unwm_filename).split(".")[0]  
        wm_filename = os.path.join(args.wm_images_dir, "{}_hidden.png".format(file_id))
        wm_for_residual_filename = os.path.join(args.wm_images_for_residuals_dir, "{}_hidden.png".format(file_id))

        wm_img = cv2.imread(wm_filename)
        wm_for_residual = cv2.imread(wm_for_residual_filename)
        unwm_img = cv2.imread(unwm_filename)

        if wm_img.shape[0] < unwm_img.shape[0]:
            unwm_img = cv2.resize(unwm_img, (wm_img.shape[0], wm_img.shape[1]), cv2.INTER_AREA)
        elif wm_img.shape[0] > unwm_img.shape[0]:
            wm_img = cv2.resize(wm_img, (unwm_img.shape[0], unwm_img.shape[1]), cv2.INTER_AREA)

        cv2.imwrite(
            os.path.join(output_dir, f"{file_id}_attacked.png"),
            attack(wm_img, unwm_img, args.percentile_threshold, args.blur_size)
        )


if __name__ == "__main__":
    main()
