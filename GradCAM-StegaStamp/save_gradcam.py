import glob
import numpy as np
import cv2
import os
from tqdm import tqdm
from decode_gradcam import GradCAMStegaStamp   

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str)
    parser.add_argument('--secret_size', type=int, default=100)
    parser.add_argument('--wm_images_dir', type=str)
    parser.add_argument('--output_dir', type=str)
    args = parser.parse_args()

    gradcam_model = GradCAMStegaStamp(args.model, args.secret_size)

    wm_files_list = sorted(glob.glob(args.wm_images_dir + '/*'), key=lambda x: os.path.basename(x).split(".")[0])

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    for wm_filename in tqdm(wm_files_list):
        file_id = os.path.basename(wm_filename).split("_")[0]

        # try:
        _, gradcam = gradcam_model.get_message(wm_filename, with_gradcam=True)
        np.save(os.path.join(output_dir, f"{file_id}_gradcam.npy"), gradcam)
        # except:
        #     continue


if __name__ == "__main__":
    main()
