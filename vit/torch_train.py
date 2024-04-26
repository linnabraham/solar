import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from base import BaseModel
from vit_pytorch import ViT
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


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

vit_model = DeepFlare_ViT(height=512, n_classes=2, n_passbands=7)
model = vit_model.model
print(model)

