from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

import torch
from torchvision.transforms import v2

from aarp_ml.torch.dataset import aia_euv, AIALogTransform

REPO_ROOT = Path(__file__).resolve().parents[1]
JSON_PATH = REPO_ROOT / "solar_dataset.json"
STATS_PATH = REPO_ROOT / "stats.json"

PASSBANDS = ["94", "131", "171", "193", "211", "304", "335"]

def load_stats():
    with open(STATS_PATH) as f:
        stats = json.load(f)
    means = [stats["mean"][f"channel_{i}"] for i in range(7)]
    stds  = [stats["std"][f"channel_{i}"]  for i in range(7)]
    return means, stds

def read_raw(item):
    imgs = []
    for i in range(7):
        with fits.open(item[str(i)], memmap=True) as hdul:
            imgs.append(hdul[0].data.astype(np.float32))
    return np.stack(imgs)

def plot(raw, model_input):
    fig, axes = plt.subplots(2, 7, figsize=(18, 5))

    for i in range(7):
        axes[0, i].imshow(raw[i], origin="lower", cmap="gray")
        axes[0, i].set_title(PASSBANDS[i])
        axes[0, i].axis("off")

        axes[1, i].imshow(model_input[i], origin="lower", cmap="viridis")
        axes[1, i].axis("off")

    axes[0, 0].set_ylabel("Raw FITS")
    axes[1, 0].set_ylabel("Model input")

    plt.tight_layout()
    plt.show()

def main():
    means, stds = load_stats()

    transform = v2.Compose([
        AIALogTransform(means, stds)
    ])

    ds = aia_euv(JSON_PATH, subset="validation", transform=transform)

    x, y = ds[0]
    raw = read_raw(ds.data[0])

    print("Label:", y.item())
    print("Raw min/max:", raw.min(), raw.max())
    print("Model input min/max:", x.min().item(), x.max().item())

    plot(raw, x.numpy())

if __name__ == "__main__":
    main()
