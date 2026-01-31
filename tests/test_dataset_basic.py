from pathlib import Path
import json
from collections import Counter

import torch
from torch.utils.data import DataLoader

from aarp_ml.torch.dataset import aia_euv, AIALogTransform
from src.torch.vit.train import get_weighted_sampler   # reuse YOUR code

REPO_ROOT = Path(__file__).resolve().parents[1]

JSON_PATH = REPO_ROOT / "solar_dataset.json"
STATS_PATH = REPO_ROOT / "stats.json"

def load_stats():
    with open(STATS_PATH) as f:
        stats = json.load(f)
    means = [stats["mean"][f"channel_{i}"] for i in range(7)]
    stds  = [stats["std"][f"channel_{i}"]  for i in range(7)]
    return means, stds

def main():
    means, stds = load_stats()

    train_ds = aia_euv(
        JSON_PATH,
        subset="training",
        transform=AIALogTransform(means, stds)
    )

    print("Train length:", len(train_ds))

    label_counts = Counter(train_ds.labels)
    print("Label distribution:", label_counts)

    sampler = get_weighted_sampler(train_ds)
    loader = DataLoader(train_ds, batch_size=4, sampler=sampler)

    x, y = next(iter(loader))
    print("Batch x shape:", x.shape)  # (B, 7, H, W)
    print("Batch y:", y.tolist())

    assert x.ndim == 4
    assert x.shape[1] == 7
    assert torch.isfinite(x).all()

    print("✓ Dataset loading test passed")

if __name__ == "__main__":
    main()
