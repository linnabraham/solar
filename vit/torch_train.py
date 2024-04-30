import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from base import BaseModel
from vit_pytorch import ViT
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch.utils.data import Dataset, DataLoader
import json
from astropy.io import fits

class aia_euv(Dataset):
    def __init__(self, json_path, subset):
        self.data = self._load_data(json_path, subset)

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
        label = torch.tensor(label, dtype=torch.long)
        return features, label

    def _read_fits_file(self, file_path):
        with fits.open(file_path) as hdul:
            data = hdul[0].data
        return data

class DeepFlare_ViT(BaseModel):
    def __init__(self,**kwargs):
        super(DeepFlare_ViT,self).__init__(**kwargs)
        n_passbands = kwargs.pop('n_passbands',None)
        height = kwargs.pop('height',None)
        n_classes = kwargs.pop('n_classes',None)
        dropout_prob = kwargs.pop('dropout',0.3)
        if n_passbands is None or height is None or n_classes is None:
            raise ValueError("Number of input passbands (filters) must be given!")
        self.model = ViT(image_size=height,patch_size=64,num_classes=n_classes,
                         dim=1024,depth=4,heads=16,channels=n_passbands,mlp_dim=9,
                         dropout=dropout_prob,emb_dropout=dropout_prob,pool="mean")

    train_dataset = aia_euv('../solar_dataset.json', subset='training')
    validation_dataset = aia_euv('../solar_dataset.json', subset='validation')
    test_dataset = aia_euv('../solar_dataset.json', subset='test')

    for i in range(len(train_dataset)):
        features, label = train_dataset[i]
        #print("Features:", features)
        #print("Label:", label)
        print(features.shape)
        break

    print("train size", len(train_dataset))
    print("validation size", len(validation_dataset))
    print("test size", len(test_dataset))

    train_loader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle=True)
    val_loader = DataLoader(validation_dataset, batch_size = args.batch_size, shuffle=False)


vit_model = DeepFlare_ViT(height=512, n_classes=2, n_passbands=7)
model = vit_model.model
print(model)

