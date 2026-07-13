import sys, os
import argparse
import wandb
from tqdm import tqdm
from torchvision.transforms import v2
import json
import pickle
import torch
import torch.nn as nn
import torchvision.models
from typing import Optional, List
from dataclasses import dataclass
import math
from torch.utils.data import DataLoader
from src.torch.vit.train import print_config, get_weighted_sampler, _run_epoch
from aarp_ml.torch.dataset import aia_euv, AIALogTransform
from aarp_ml.torch.model import SaveBestModel
from aarp_ml.dataset import all_wavelengths

def modify_alexnet(model, in_channels=7):
    """
    This function modifies the base architecture of AlexNet to conform as far as
    posssible with the architecture that used in tensorflow
    """

    model.features[0] = nn.Conv2d(
        in_channels=in_channels,
        out_channels=96,
        kernel_size=(5, 5),
        stride=(2, 2),
    )

    model.features[3] = nn.Conv2d(96, 256, kernel_size=(5, 5), stride=(1, 1), padding='same')
    model.features[6] = nn.Conv2d(256, 384, kernel_size=(3, 3), stride=(1, 1), padding='same')
    model.features[8] = nn.Conv2d(384, 384, kernel_size=(3, 3), stride=(1, 1), padding='same')
    model.features[10] = nn.Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding='same')

    #TODO: find out why the following code doesn't work
    # model.classifier[6].out_features = 2

    model.classifier[6] = nn.Linear(in_features=4096, out_features=2)

    # Append sigmoid to convert logits to probability (e.g., for binary classification)
    model = nn.Sequential(model, nn.Sigmoid())

    return model

@dataclass
class TrainingConfig:
    # Required parameters
    json_path: str
    stats_file: str

    # Optional training parameters
    batch_size: int = 32
    epochs: int = 250
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
    channel_indices: Optional[list] = None  # indices into AIA_CHANNELS wavelength order; None = all 7

    # System parameters
    device: str = "cuda:0"
    memory_threshold: int = 5000  # GPU memory threshold measured in megabytes

    def __post_init__(self):
        if self.channel_indices is None:
            self.channel_indices = list(range(self.n_channels))
        else:
            self.n_channels = len(self.channel_indices)

def init_data(config):
    with open(config.json_path, 'r') as json_file:
        metadata = json.load(json_file)

    with open(config.stats_file, 'rb') as pickle_file:
        stats_data = pickle.load(pickle_file)

    means = [stats_data.get('mean').get(f'channel_{i}') for i in config.channel_indices]
    stds = [stats_data.get('std').get(f'channel_{i}') for i in config.channel_indices]

    transform = AIALogTransform(means=means, stds=stds)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Create datasets
    train_dataset = aia_euv(
        config.json_path,
        subset='training',
        transform=v2.Compose([
            AIALogTransform(means, stds),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5)
        ]),
        channel_indices=config.channel_indices
    )
    val_dataset = aia_euv(
        config.json_path,
        subset='validation',
        transform=v2.Compose([AIALogTransform(means, stds)]),
        channel_indices=config.channel_indices
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        sampler=get_weighted_sampler(train_dataset)
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=True
    )
    return metadata, transform, device, train_loader, val_loader

def train(config: TrainingConfig, model):
    """Main training function."""
    # Initialize wandb
    wandb.init(project="flare_torch", config=vars(config))

    # Print configuration
    print_config(config)
    metadata, transform, device, train_loader, val_loader = init_data(config)
    model = model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
        if config.scheduler_type == "cosine_annealing" else None
    )

    # Create output directory
    output_dir = os.path.join("output", wandb.run.name)
    os.makedirs(output_dir, exist_ok=True)

    # Setup model saving
    save_best_model = SaveBestModel(monitor='val_loss', mode='min')

    # Load checkpoint if retraining
    start_epoch = 0
    if config.retrain and config.trained_model_path:
        checkpoint = torch.load(config.trained_model_path, map_location=device)
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint.get('epoch', -1) + 1
        else:
            model.load_state_dict(checkpoint)

    # Training loop
    torch.cuda.reset_peak_memory_stats()
    n_steps_per_epoch = math.ceil(len(train_loader.dataset) / config.batch_size)

    for epoch in range(start_epoch, config.epochs):
        success = _run_epoch(
            epoch=epoch,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
            n_steps_per_epoch=n_steps_per_epoch,
            save_callback=save_best_model,
            output_dir=output_dir
        )
        if not success:
            break

def parse_args() -> TrainingConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-path", default="solar_dataset.json",
                       help="Path to JSON file containing dataset information")
    parser.add_argument("--stats-file", default="stats.pkl",
                       help="Path to statistics file containing means and stds")
    parser.add_argument("--batch-size", type=int, default=TrainingConfig.batch_size)
    parser.add_argument("--epochs", type=int, default=TrainingConfig.epochs)
    parser.add_argument("--lr", type=float, default=TrainingConfig.learning_rate)
    parser.add_argument("--channels", type=int, nargs="+", default=None,
                       help="AIA passbands to use, e.g. --channels 94 131. "
                            f"Choices: {all_wavelengths}. Default: all 7, in wavelength order.")

    args = parser.parse_args()

    channel_indices = None
    if args.channels is not None:
        unknown = [c for c in args.channels if c not in all_wavelengths]
        if unknown:
            parser.error(f"Unknown channel(s) {unknown}. Choices: {all_wavelengths}")
        channel_indices = [all_wavelengths.index(c) for c in args.channels]

    return TrainingConfig(
        json_path=args.json_path,
        stats_file=args.stats_file,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        channel_indices=channel_indices,
    )

if __name__=="__main__":
    config = parse_args()
    alexnet = torchvision.models.alexnet()
    model = modify_alexnet(alexnet, in_channels=config.n_channels)
    train(config, model)
