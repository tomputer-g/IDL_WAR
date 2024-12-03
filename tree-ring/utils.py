import matplotlib.pyplot as plt
import torch

def visualize_tensor(tensor, name="./tmp.png"):
    # source: https://github.com/YuxinWenRick/tree-ring-watermark/issues/19#issuecomment-1894069738
    channels = tensor.squeeze(0).cpu().numpy()
    fig, axs = plt.subplots(1, 4, figsize=(12, 3))

    for i, ax in enumerate(axs):
        ax.imshow(channels[i], cmap="gray")
        ax.set_title(f"Channel {i+1}")
        ax.axis("off")

    plt.savefig(name)
    plt.show()
    plt.close()

# perform fast fourier transform on latents
# use fftshift to move lower frequency components to the center
# where we will put our watermark
def fft(latent_tensor):
    return torch.fft.fftshift(torch.fft.fft2(latent_tensor), dim=(-1, -2))


# inverse fast fourier transform on latents
def ifft(fft_latent_tensor):
    return torch.fft.ifft2(torch.fft.ifftshift(fft_latent_tensor, dim=(-1, -2)))


def visualize_latent(latents, fft_applied=False, name="tmp"):
    if not fft_applied:
        # apply fft
        latents = fft(latents)

    # useful code for debugging if injected key is reasonable
    visualize_tensor(latents.real, name=f"{name}_real.png")
    visualize_tensor(latents.imag, name=f"{name}_imag.png")