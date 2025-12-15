from torch.utils.data import Dataset
from astropy.io import fits
import torch
import json
import numpy as np

class AIALogTransform:
    def __init__(self, means, stds):
        self.means = torch.tensor(means).view(-1, 1, 1)  # Shape (7, 1, 1) for broadcasting
        self.stds = torch.tensor(stds).view(-1, 1, 1)    # Shape (7, 1, 1) for broadcasting

    def __call__(self, x):
        x[x < 0] = 0
        x[x == 0] = 1
        x = torch.log(x)
        x = (x - self.means.to(x.device)) / self.stds.to(x.device)  # Z-score normalization
        return x

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
