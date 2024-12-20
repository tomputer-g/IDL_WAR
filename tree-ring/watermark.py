import torch
from scipy.stats import ncx2
import os

from utils import visualize_tensor, fft, ifft


# add watermark to noise, returns key
def watermark(
    latent_tensor,
    type="rings",
    radius=10,
    device=None,
    channel=0,
    key=None,
    visualize_key=False,
    visualization_folder="",
    visualization_name="",
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # apply fft
    fft_applied = fft(latent_tensor)

    # generate key
    key, mask = generate_key(
        size=latent_tensor.shape[-1],
        type=type,
        radius=radius,
        device=device,
        dtype=fft_applied.dtype,
        key=key
    )

    # inject key
    fft_applied[channel, mask] = key[mask]

    # useful code for debugging if injected key is reasonable
    if visualize_key:
        visualize_tensor(fft_applied.real, name=os.path.join(visualization_folder, f"{visualization_name}_real.png"))
        visualize_tensor(fft_applied.imag, name=os.path.join(visualization_folder, f"{visualization_name}_imag.png"))

    # reverse fft
    latent_tensor = ifft(fft_applied).real

    # return key, mask, and new latent
    return latent_tensor, key, mask

# generate the 2D watermark key values
def generate_key(
    size,
    type="rings",
    radius=10,
    device=None,
    dtype=torch.complex64,
    key=None
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    height, width = size, size

    circle_mask = torch.zeros((height, width), dtype=bool, device=device)
    y, x = torch.meshgrid(
        torch.arange(height, device=device),
        torch.arange(width, device=device),
        indexing="ij",
    )

    center_x = (width - 1) / 2
    center_y = (height - 1) / 2

    distance_from_center_squared = torch.square(x - center_x) + torch.square(
        y - center_y
    )

    tensor_radius = torch.tensor([radius], device=device)
    circle_mask = distance_from_center_squared <= torch.square(tensor_radius)

    if key is None:
        key = torch.zeros_like(circle_mask, dtype=dtype, device=device)

    if key.shape != circle_mask.shape:
        raise Exception("Key shape is not the same as the mask shape")

    match type:
        case "zeros":
            key[circle_mask] = 0
        case "rand":
            key[circle_mask] = fft(torch.randn(circle_mask.shape, device=device))[
                circle_mask
            ]
        case "rings":
            values = fft(torch.randn((size, size)))

            for i in range(1, radius + 1):
                inner_radius_tensor = torch.tensor([i - 1], device=device)
                outer_radius_tensor = torch.tensor([i], device=device)

                if i == 1:
                    # handle separately to include center bit
                    in_ring_mask = distance_from_center_squared <= torch.square(
                        outer_radius_tensor
                    )
                else:
                    in_ring_mask = (
                        torch.square(inner_radius_tensor) < distance_from_center_squared
                    ) & (
                        distance_from_center_squared
                        <= torch.square(outer_radius_tensor)
                    )
                key[in_ring_mask] = values[0][i - 1 + int(center_x)]
        case _:
            raise Exception(f"Invalid tree-ring type {type}")

    return key, circle_mask

def extract_key(latent_tensor, mask, channel=0):
    # apply fft
    fft_applied = fft(latent_tensor)

    # extract bits corresponding to key
    key_from_latents = fft_applied[channel, mask]  # assume key is 0th dim

    # convert to larger dtype first so we don't get inf
    key_from_latents = key_from_latents.type(torch.complex64)

    return key_from_latents


def p_value(
    latent_tensor,
    key,
    mask,
    channel=0,
    visualize_renoised=False,
    visualization_folder="",
    visualization_name="",
):
    # apply fft
    fft_applied = fft(latent_tensor)

    # useful code for debugging if injected key is reasonable
    if visualize_renoised:
        visualize_tensor(fft_applied.real, name=f"{visualization_name}_renoised_real.png")
        visualize_tensor(fft_applied.imag, name=f"{visualization_name}_renoised_imag.png")

    # extract bits corresponding to key
    key_from_latents = fft_applied[channel, mask]

    # convert to larger dtype first so we don't get inf
    key_from_latents = key_from_latents.type(torch.complex64)
    key = key.type(torch.complex64)

    # calculate score
    variance = 1 / mask.sum() * torch.square(torch.abs(key_from_latents)).sum()
    dof = mask.sum()
    nc = 1 / variance * torch.square(torch.abs(key[mask])).sum()
    score = 1 / variance * torch.square(torch.abs(key[mask] - key_from_latents)).sum()

    # calculate p-value
    p_val = ncx2.cdf(score.cpu(), df=dof.cpu(), nc=nc.cpu())

    return p_val

def l1_dist(
    latent_tensor,
    key,
    mask,
    channel=0,
    visualize_renoised=False,
    visualization_folder="",
    visualization_name="",
):
    # apply fft
    fft_applied = fft(latent_tensor)

    # useful code for debugging if injected key is reasonable
    if visualize_renoised:
        visualize_tensor(fft_applied.real, name=f"{visualization_name}_renoised_real.png")
        visualize_tensor(fft_applied.imag, name=f"{visualization_name}_renoised_imag.png")


    # extract bits corresponding to key
    key_from_latents = fft_applied[channel, mask]

    # convert to larger dtype first so we don't get inf
    key_from_latents = key_from_latents.type(torch.complex64)
    key = key.type(torch.complex64)

    # calculate dist
    dist = torch.abs(key[mask] - key_from_latents).mean().item()

    return dist

def detect_pval(
    latent_tensor,
    key,
    mask,
    p_val_thresh=0.01,
    visualize_renoised=False,
    visualization_folder="",
    visualization_name="",
) -> tuple[float, bool]:
    p_val = p_value(
        latent_tensor,
        key,
        mask,
        visualize_renoised=visualize_renoised,
        visualization_folder=visualization_folder,
        visualization_name=visualization_name,
    )
    return float(p_val), bool(p_val < p_val_thresh)

def detect_dist(latent_tensor, key, mask, dist_thresh=77) -> tuple[float, bool]:
    dist = l1_dist(
        latent_tensor,
        key,
        mask,
        visualize_renoised=visualize_renoised,
        visualization_folder=visualization_folder,
        visualization_name=visualization_name,
    )
    return float(dist), bool(dist < dist_thresh)
