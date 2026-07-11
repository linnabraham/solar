from torch.utils.data import Dataset, DataLoader
from astropy.io import fits
import torch
import json
import numpy as np
from tqdm import tqdm

class AIALogTransform:
    def __init__(self, means, stds):
        self.log_means = torch.tensor(means).view(-1, 1, 1)  # Shape (7, 1, 1) for broadcasting; must be log-space
        self.log_stds = torch.tensor(stds).view(-1, 1, 1)    # Shape (7, 1, 1) for broadcasting; must be log-space

    def __call__(self, x):
        x = x.clamp(min=1)  # AIA pixels are integer DN; clamp so log is defined and ≥ 0
        x = torch.log(x)
        x = (x - self.log_means.to(x.device)) / self.log_stds.to(x.device)  # Z-score normalization
        return x


def compute_log_mean_and_std(json_path, batch_size=32, num_workers=4):
    """Compute per-channel mean and std of log-transformed pixel values on the training set.

    Returns log-space statistics so that AIALogTransform produces a proper z-score:
        (log(x) - log_mean) / log_std
    """
    dataset = aia_euv(json_path=json_path, subset='training', transform=None)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    n_channels = 7
    sum_log    = torch.zeros(n_channels)
    sum_log_sq = torch.zeros(n_channels)
    count = 0

    for features, _ in tqdm(loader, desc="Computing log-space stats"):
        log_f = torch.log(features.clamp(min=1))   # [B, C, H, W]
        sum_log    += log_f.sum(dim=[0, 2, 3])
        sum_log_sq += (log_f ** 2).sum(dim=[0, 2, 3])
        count += features.shape[0] * features.shape[2] * features.shape[3]

    mean = sum_log / count
    std  = torch.sqrt(sum_log_sq / count - mean ** 2)
    return mean, std


def compute_raw_mean_and_std(json_path, batch_size=32, num_workers=4):
    """Compute per-channel mean and std of raw (linear-space) pixel values on the training set.

    Reproduces the pre-fix (buggy) stats: these are meant to be paired with
    AIALogTransform, which log-transforms pixels before z-scoring, so using
    these stats there is the historical mean/std-space mismatch, not a fix.
    """
    dataset = aia_euv(json_path=json_path, subset='training', transform=None)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    n_channels = 7
    sum_raw    = torch.zeros(n_channels)
    sum_raw_sq = torch.zeros(n_channels)
    count = 0

    for features, _ in tqdm(loader, desc="Computing raw-space stats"):
        sum_raw    += features.sum(dim=[0, 2, 3])
        sum_raw_sq += (features ** 2).sum(dim=[0, 2, 3])
        count += features.shape[0] * features.shape[2] * features.shape[3]

    mean = sum_raw / count
    std  = torch.sqrt(sum_raw_sq / count - mean ** 2)
    return mean, std


# Un-normalize and inverse log-transform function
def inverse_transform(image_tensor, means, stds):
    """
    Inverts the transformation: (ln(x) - mean) / std.
    The inverse is: exp((x_norm * std) + mean).
    """
    # image_tensor shape: [1, C, H, W]

    # 1. Move to CPU and remove the batch dimension
    image_numpy = image_tensor.squeeze(0).cpu().numpy() # Shape: [C, H, W]

    # 2. Reshape means and stds to [C, 1, 1] for broadcasting
    # Note: means and stds here are for the natural log-space.
    means_np = np.array(means)[:, None, None]
    stds_np = np.array(stds)[:, None, None]

    # Step 1: Inverse standardization (un-normalize)
    # x_log = (x_norm * std) + mean (This is now the natural log of the intensity)
    x_log = (image_numpy * stds_np) + means_np

    # Step 2: Inverse natural log transformation: exp(x_log)
    original_intensity = np.exp(x_log)

    return original_intensity # Shape: [C, H, W]

class aia_euv(Dataset):
    def __init__(self, json_path, subset, transform=None):
        self.data = self._load_data(json_path, subset)
        self.transform = transform
        self.labels = [item['label'] for item in self.data]

    def _load_data(self, json_file, subset):
        with open(json_file, 'r') as f:
            data_dict = json.load(f)
        return data_dict.get(subset, [])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        features = [self._read_fits_file(item[str(i)]) for i in range(7)]  # Read FITS files
        features = np.stack(features, axis=0)  # Stack along a new axis
        label = item['label']
        features = torch.tensor(features, dtype=torch.float32)
        if self.transform:
            features = self.transform(features)
        label = torch.tensor(label, dtype=torch.long)
        return features, label

    def _read_fits_file(self, file_path):
        with fits.open(file_path,memmap=True) as hdul:
            data = hdul[0].data
        return data
