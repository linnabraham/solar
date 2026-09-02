import torch
import torch.nn as nn
import torchvision
from .base import BaseModel
from vit_pytorch import ViT

def build_pretrained_vit(n_channels: int = 7, n_classes: int = 2, pretrained: bool = True,
                         dropout: float = 0.0, attention_dropout: float = 0.0) -> torch.nn.Module:
    """torchvision vit_l_16, conv_proj swapped for n_channels input, heads.head swapped for
    n_classes output. pretrained=True loads ImageNet1K_V1 weights (for training from scratch);
    pretrained=False skips it (for eval/inference, about to load a fine-tuned checkpoint anyway).

    dropout/attention_dropout are forwarded to torchvision's VisionTransformer (both default to
    0.0, torchvision's own default -- previously not exposed here at all, so every run before
    2026-07-27 trained with zero dropout regardless of intent). Dropout layers carry no
    parameters, so this is fully compatible with a pretrained checkpoint's state_dict either way.

    Single source of truth for this architecture -- train_pretrained.py, test_pretrained.py, and
    evaluate_aarp.py all build from here so the reconstructed graph is guaranteed identical to
    whatever produced a given .pth file. Do NOT wrap heads.head in nn.Sequential(Dropout, Linear)
    -- that changes the state_dict key to 'heads.1.*' and breaks load_state_dict against existing
    checkpoints (see the now-superseded VIT_Pretrained in vit_pretrained.py for that variant).
    """
    weights = "IMAGENET1K_V1" if pretrained else None
    model = torchvision.models.vit_l_16(weights=weights, dropout=dropout, attention_dropout=attention_dropout)

    conv1_out = model.conv_proj.out_channels
    model.conv_proj = nn.Conv2d(n_channels, conv1_out, kernel_size=(16, 16), stride=(16, 16))

    lin_in = model.heads.head.in_features
    model.heads.head = nn.Linear(lin_in, n_classes, bias=True)

    return model

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
                         dim=1024,depth=4,heads=16,channels=n_passbands,mlp_dim=512,
                         dropout=dropout_prob,emb_dropout=dropout_prob,pool="mean")

class SaveBestModel:
    def __init__(self, monitor='val_loss', mode='min', initial_best_value=None):
        """initial_best_value: seed the tracker from a prior run's best (e.g. the val_metric
        already stored in an existing trained_model.pth) instead of always starting at
        inf/-inf. Without this, resuming training (--retrain) resets the tracker to inf, so the
        very first validation of the resumed run overwrites trained_model.pth even if it's worse
        than the pre-resume best -- silently discarding it (only recoverable if --save-all-epochs
        happened to also keep the true-best epoch's checkpoint separately)."""
        self.monitor = monitor
        self.mode = mode
        if mode == 'min':
            self.best_value = float('inf') if initial_best_value is None else float(initial_best_value)
            self.monitor_op = lambda x, y: x < y
        else:
            self.best_value = float('-inf') if initial_best_value is None else float(initial_best_value)
            self.monitor_op = lambda x, y: x > y

    def __call__(self, val_metric, model, filepath, optimizer=None, epoch=None, wandb_run_id=None):
        if self.monitor_op(val_metric, self.best_value):
            print(f"Validation {self.monitor}: {val_metric} improved from {self.best_value} to {val_metric}. Saving model...")
            self.best_value = val_metric

            checkpoint = {
                'model_state_dict': model.state_dict(),
                'val_metric': val_metric
            }

            if optimizer is not None:
                checkpoint['optimizer_state_dict'] = optimizer.state_dict()
            if epoch is not None:
                checkpoint['epoch'] = epoch
            if wandb_run_id is not None:
                checkpoint['wandb_run_id'] = wandb_run_id

            torch.save(checkpoint, filepath)
        else:
            print(f"Validation {self.monitor}: {val_metric} did not improve from {self.best_value}.")
