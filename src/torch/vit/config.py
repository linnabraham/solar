from dataclasses import dataclass
from typing import Optional

@dataclass
class TrainingConfig:
    # Required parameters
    json_path: str
    stats_file: str

    # Optional training parameters
    batch_size: int = 32
    epochs: int = 50
    learning_rate: float = 0.001
    scheduler_type: Optional[str] = None
    retrain: bool = False
    trained_model_path: Optional[str] = None
    run_name: Optional[str] = None
    use_l1: bool = False
    l1_lambda: float = 0.01
    save_all_epochs: bool = False  # also save every epoch's checkpoint (for post-hoc checkpoint-selection experiments)
    seed: Optional[int] = None  # if set, fixes torch/numpy/python RNG state for reproducible runs
    model_type: str = "deepflare_vit"  # bookkeeping only (wandb + run metadata); does not drive train.py branching
    dropout: Optional[float] = None  # only consumed by train_pretrained.py's build_model_fn; None -> library default (0.0)
    attention_dropout: Optional[float] = None  # same, for vit_l_16's attention_dropout

    # Model parameters
    image_height: int = 512
    n_classes: int = 2
    n_channels: int = 7
    channel_indices: Optional[list] = None  # indices into AIA_CHANNELS wavelength order; None = all 7

    # System parameters
    device: str = "cuda:0"
    memory_threshold: int = 5000  # GPU memory threshold measured in megabytes

    def __post_init__(self):
        if self.channel_indices is None:
            self.channel_indices = list(range(self.n_channels))
        else:
            self.n_channels = len(self.channel_indices)