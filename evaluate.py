import torch
import argparse
from pytorch_fid.fid_score import calculate_fid_given_paths


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
    gt_folder,
    unwatermarked_folder,
    watermarked_folder,
):
    # img.resize((299, 299))
    unwatermarked_fid, watermarked_fid = eval_fid(
        gt_folder, unwatermarked_folder, watermarked_folder
    )
    print(f"Unwatermarked FID: {unwatermarked_fid}")
    print(f"Watermarked FID: {watermarked_fid}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_folder', type=str, default=None)
    parser.add_argument('--unwatermarked_folder', type=str, default=None)
    parser.add_argument('--watermarked_folder', type=str, default=None)
    args = parser.parse_args()

    main(args.gt_folder, args.unwatermarked_folder, args.watermarked_folder)