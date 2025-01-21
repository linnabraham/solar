import torch
from .base import BaseModel
from vit_pytorch import ViT

class DeepFlare_ViT(BaseModel):
    def __init__(self,**kwargs):
        super(DeepFlare_ViT,self).__init__(**kwargs)
        n_passbands = kwargs.pop('n_passbands',None)
        height = kwargs.pop('height',None)
        n_classes = kwargs.pop('n_classes',None)
        dropout_prob = kwargs.pop('dropout',0.3)
        if n_passbands is None or height is None or n_classes is None:
            raise ValueError("Number of input passbands (filters) must be given!")
        self.model = ViT(image_size=height,patch_size=16,num_classes=n_classes,
                         dim=1024,depth=4,heads=16,channels=n_passbands,mlp_dim=9,
                         dropout=dropout_prob,emb_dropout=dropout_prob,pool="mean")

class SaveBestModel:
    def __init__(self, monitor='val_loss', mode='min'):
        self.monitor = monitor
        self.mode = mode
        if mode == 'min':
            self.best_value = float('inf')
            self.monitor_op = lambda x, y: x < y
        else:
            self.best_value = float('-inf')
            self.monitor_op = lambda x, y: x > y

    def __call__(self, val_metric, model, filepath):
        if self.monitor_op(val_metric, self.best_value):
            print(f"Validation {self.monitor}: {val_metric} improved from {self.best_value} to {val_metric}. Saving model...")
            self.best_value = val_metric
            torch.save(model.state_dict(), filepath)
        else:
            print(f"Validation {self.monitor}: {val_metric} did not improve from {self.best_value}.")
