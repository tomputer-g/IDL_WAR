import torch

# add watermark to noise, returns key
def watermark(latent_tensor, type="rings", radius=10):
    # apply fft
    fft_applied = fft(latent_tensor)
    
    # generate key
    key, mask = generate_key(type=type, radius=radius)

    # inject key
    center_x = (fft_applied.shape[2] - 1) // 2
    center_y = (fft_applied.shape[1] - 1) // 2
    for dim in range(fft_applied.shape[0]):
        fft_applied[dim, center_y-radius:center_y+radius, center_x-radius:center_x+radius][mask] = key[mask]

    # reverse fft
    latent_tensor = ifft(fft_applied).real

    # enlarge mask/key to size of image
    large_key = torch.zeros(fft_applied.shape[1:], dtype=torch.complex64)
    large_key[center_y-radius:center_y+radius, center_x-radius:center_x+radius][mask] = key[mask]

    large_mask = torch.zeros(fft_applied.shape[1:], dtype=bool)
    large_mask[center_y-radius:center_y+radius, center_x-radius:center_x+radius] = mask

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
def generate_key(type="rings", radius=10):
    height, width = radius * 2, radius * 2

    circle_mask = torch.zeros((height, width), dtype=bool)
    y, x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")

    center_x = (width - 1) / 2
    center_y = (height - 1) / 2

    distance_from_center_squared = (
        torch.square(x - center_x)
        + torch.square(y - center_y)
    )

    tensor_radius = torch.tensor([radius])
    circle_mask = distance_from_center_squared <= torch.square(tensor_radius)
    key = torch.zeros_like(circle_mask, dtype=torch.float)

    match type:
        case "zeros":
            key[circle_mask] = 0
        case "rand":
            key[circle_mask] = torch.randn(circle_mask.shape)[circle_mask]
        case "rings":
            values = torch.randn(radius)
            
            for i in range(1, radius+1):
                inner_radius_tensor = torch.tensor([i-1])
                outer_radius_tensor = torch.tensor([i])

                if i == 1:
                    # handle separately to include center bit
                    in_ring_mask = distance_from_center_squared <= torch.square(outer_radius_tensor)
                else:
                    in_ring_mask = (torch.square(inner_radius_tensor) < distance_from_center_squared) \
                                    & (distance_from_center_squared <= torch.square(outer_radius_tensor))
                key[in_ring_mask] = values[i-1]
        case _:
            raise Exception(f"Invalid tree-ring type {type}")        
    
    key = fft(key)

    return key, circle_mask