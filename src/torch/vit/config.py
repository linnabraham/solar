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
    use_l1: bool = False
    l1_lambda: float = 0.01

    # Model parameters
    image_height: int = 512
    n_classes: int = 2
    n_channels: int = 7

    # System parameters
    device: str = "cuda:0"
    memory_threshold: int = 8000  # GPU memory threshold measured in megabytes
