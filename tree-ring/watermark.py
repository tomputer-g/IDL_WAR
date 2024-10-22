import torch
from scipy.stats import ncx2

from utils import visualize_tensor


# add watermark to noise, returns key
def watermark(latent_tensor, type="rings", radius=10, device="cpu"):
    # apply fft
    fft_applied = fft(latent_tensor)

    # generate key
    key, mask = generate_key(
        size=latent_tensor.shape[-1], type=type, radius=radius, device=device
    )

    # enlarge mask/key to size of image
    center_x = (fft_applied.shape[2] - 1) // 2
    center_y = (fft_applied.shape[1] - 1) // 2
    large_key = torch.zeros(fft_applied.shape[1:], dtype=torch.complex64, device=device)
    large_key[
        center_y - radius : center_y + radius, center_x - radius : center_x + radius
    ][mask] = key[mask]

    large_mask = torch.zeros(fft_applied.shape[1:], dtype=bool, device=device)
    large_mask[
        center_y - radius : center_y + radius, center_x - radius : center_x + radius
    ] = mask

    # inject key
    fft_applied[0, large_mask] = large_key[large_mask]

    # useful code for debugging if injected key is reasonable
    # visualize_tensor(fft_applied.real, name="real.png")
    # visualize_tensor(fft_applied.imag, name="imag.png")

    # reverse fft
    latent_tensor = ifft(fft_applied).real

    # return key, mask, and new latent
    return latent_tensor, large_key, large_mask


# perform fast fourier transform on latents
# use fftshift to move lower frequency components to the center
# where we will put our watermark
def fft(latent_tensor):
    return torch.fft.fftshift(torch.fft.fft2(latent_tensor), dim=(-1, -2))


# inverse fast fourier transform on latents
def ifft(fft_latent_tensor):
    return torch.fft.ifft2(torch.fft.ifftshift(fft_latent_tensor, dim=(-1, -2)))


# generate the 2D watermark key values
def generate_key(size, type="rings", radius=10, device="cpu"):
    height, width = radius * 2, radius * 2

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
    key = torch.zeros_like(circle_mask, dtype=torch.complex64, device=device)

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


def p_value(latent_tensor, key, mask):
    # apply fft
    fft_applied = fft(latent_tensor)

    # extract bits corresponding to key
    key_from_latents = fft_applied[0, mask]  # assume key is 0th dim

    # calculate score
    variance = 1 / mask.sum() * torch.square(torch.abs(key_from_latents)).sum()
    dof = mask.sum()
    nc = 1 / variance * torch.square(torch.abs(key[mask])).sum()
    score = 1 / variance * torch.square(torch.abs(key[mask] - key_from_latents)).sum()

    # calculate p-value
    p_val = ncx2.cdf(score.cpu(), df=dof.cpu(), nc=nc.cpu())

    return p_val


def detect(latent_tensor, key, mask, p_val_thresh=0.01):
    return p_value(latent_tensor, key, mask) < p_val_thresh
