import matplotlib.pyplot as plt


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
