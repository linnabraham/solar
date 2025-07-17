import torch
import json
from tqdm import tqdm
import pickle
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from aarp_ml.torch.dataset import aia_euv, AIALogTransform
from src.torch.vit.train import TrainingConfig
from torch.utils.data import Subset

def save_stats(mean, std, out_path, write_json=False):
    stats = {
        'mean': {f'channel_{i}': v.item() for i, v in enumerate(mean)},
        'std': {f'channel_{i}': v.item() for i, v in enumerate(std)},
    }
    if write_json:
        out_path = f"{out_path}.json"
        with open(out_path,'w') as f:
            json.dump(stats,f,indent=4)
    else:
        out_path = f"{out_path}.pkl"
        with open(out_path, 'wb') as f:
            pickle.dump(stats, f)
    print(f"Saved stats to {out_path}")

def compute_mean_std(dataloader):
    n_pixels = 0
    sum_ = None
    sum_sq = None

    for inputs, _ in tqdm(dataloader, desc="Computing mean/std"):
        inputs = inputs.to(torch.float64)  # [B, 7, H, W]
        B, C, H, W = inputs.shape
        pixels = B * H * W

        if sum_ is None:
            sum_ = torch.zeros(C, dtype=torch.float64)
            sum_sq = torch.zeros(C, dtype=torch.float64)

        sum_ += inputs.sum(dim=(0, 2, 3))  # sum over batch and spatial dims
        sum_sq += (inputs ** 2).sum(dim=(0, 2, 3))
        n_pixels += pixels

    mean = sum_ / n_pixels
    std = torch.sqrt((sum_sq / n_pixels) - mean ** 2)
    return mean.float(), std.float()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--subset")
    args = parser.parse_args(['--batch_size','16','--num_workers','16',
                              '--subset','training'])
    print(args)

    config = TrainingConfig(json_path="solar_dataset.json", stats_file="stats.pkl")

    dataset = aia_euv(config.json_path,
                      subset=args.subset,
                      )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True
    )

    mean, std = compute_mean_std(dataloader)

    print("\nMean per passband:", mean.tolist())
    print("Std per passband: ", std.tolist())

    save_stats(mean, std, "stats", write_json=True)
