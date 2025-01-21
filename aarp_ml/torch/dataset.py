from torch.utils.data import Dataset
from astropy.io import fits
import torch
import json
import numpy as np

class CustomTransform:
    def __init__(self, means, stds):
        self.means = torch.tensor(means).view(-1, 1, 1)  # Shape (7, 1, 1) for broadcasting
        self.stds = torch.tensor(stds).view(-1, 1, 1)    # Shape (7, 1, 1) for broadcasting

    def __call__(self, x):
        x[x < 0] = 0
        x[x == 0] = 1
        x = torch.log(x)
        return x

class aia_euv(Dataset):
    def __init__(self, json_path, subset, transform=None):
        self.data = self._load_data(json_path, subset)
        self.transform = transform

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
        with fits.open(file_path) as hdul:
            data = hdul[0].data
        return data
