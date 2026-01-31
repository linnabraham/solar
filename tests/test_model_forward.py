from pathlib import Path
import json
import torch

from torchvision.transforms import v2
from aarp_ml.torch.dataset import aia_euv, AIALogTransform
from aarp_ml.torch.model import DeepFlare_ViT

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

    ds = aia_euv(
        JSON_PATH,
        subset="validation",
        transform=v2.Compose([AIALogTransform(means, stds)])
    )

    model = DeepFlare_ViT(
        height=ds[0][0].shape[-1],
        n_classes=2,
        n_passbands=7
    ).model

    x, y = ds[0]
    x = x.unsqueeze(0)  # batch dimension

    with torch.no_grad():
        logits = model(x)

    print("Logits:", logits)
    print("Shape:", logits.shape)

    assert logits.shape == (1, 2)
    print("✓ Forward pass test passed")

if __name__ == "__main__":
    main()
