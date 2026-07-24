import os
import time
import argparse
import math
import random
from typing import Callable, Optional
import wandb
from tqdm import tqdm
import pickle
import numpy as np
from torchvision.transforms import v2
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from collections import Counter, defaultdict
from sklearn.metrics import confusion_matrix
from aarp_ml.torch.dataset import aia_euv, AIALogTransform
from aarp_ml.torch.model import DeepFlare_ViT, SaveBestModel
from aarp_ml.dataset import all_wavelengths
from src.torch.vit.config import TrainingConfig

def set_seed(seed: int):
    """Fix torch/numpy/python RNG state so a run is reproducible.

    Must be called before model instantiation (weight init) and before building
    the weighted sampler / random-flip transforms, so every downstream random
    draw is deterministic given the same seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_weighted_sampler(dataset) -> WeightedRandomSampler:
    """Create a sampler that balances by AARP identity, not just raw image/class frequency.

    Plain inverse-class-frequency weighting (image-count based) balances the class ratio but
    implicitly over-samples whichever AARPs happen to have the most frames -- e.g. in fold_0's
    training split, positive AARPs range from 110 to 627 frames, a 5.7x spread, so the old
    scheme drew the highest-frame-count AARPs far more often, both in aggregate and within
    individual batches. That's a plausible contributor to the region-identity-memorization
    pattern documented in project_region_count_generalization_problem and the batch-homogeneity
    risk discussed with the user.

    This scheme instead gives every AARP the same total sampling mass within its class (split
    evenly across that AARP's own frames), and keeps the same overall class balance as before
    (each class gets equal total mass). With N classes and n_c distinct AARPs in class c, each
    frame of an AARP with f frames gets weight (1/N) / n_c / f -- generalizing the old per-class
    scheme down one level of granularity, from "class" to "AARP within class".
    """
    aarp_ids = [item['aarp_id'] for item in dataset.data]
    labels = dataset.labels

    aarps_by_class = defaultdict(set)
    frames_per_aarp = Counter()
    for aarp_id, label in zip(aarp_ids, labels):
        aarps_by_class[label].add(aarp_id)
        frames_per_aarp[aarp_id] += 1

    n_classes = len(aarps_by_class)
    sample_weights = [
        (1.0 / n_classes) / len(aarps_by_class[label]) / frames_per_aarp[aarp_id]
        for aarp_id, label in zip(aarp_ids, labels)
    ]
    return WeightedRandomSampler(weights=sample_weights, num_samples=len(dataset), replacement=True)

def validate_model(*, model, val_dl, loss_func, device):
    """Validate model and compute metrics."""
    model.eval()
    val_loss = 0.
    y_true, y_pred = [], []

    with torch.inference_mode():
        for images, labels in tqdm(val_dl, leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            val_loss += loss_func(outputs, labels) * labels.size(0)
            pred_scores, predicted = torch.max(outputs.data, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

        TN, FP, FN, TP = confusion_matrix(y_true, y_pred).ravel()
        precision = TP/(TP+FP) if (TP+FP) > 0 else 0
        recall = TP/(TP+FN) if (TP+FN) > 0 else 0

        # Print confusion matrix metrics
        print(f"\nConfusion Matrix Stats:")
        print(f"TP: {TP}, FP: {FP}")
        print(f"FN: {FN}, TN: {TN}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}\n")

    return val_loss / len(val_dl.dataset), (TP+TN) / len(val_dl.dataset), precision, recall

def _run_epoch(*, epoch, model, train_loader, val_loader, criterion, 
               optimizer, scheduler, config: TrainingConfig, n_steps_per_epoch,
               save_callback, output_dir):
    """Run a single training epoch."""
    model.train()
    epoch_train_loss = 0.0
    num_batches = 0
    example_ct = 0
    start_time = time.time()

    print(f"Epoch: {epoch}")

    for step, (inputs, labels) in tqdm(enumerate(train_loader), total=len(train_loader), leave=False):
        # Check memory usage
        if torch.cuda.memory_allocated() / (1024 ** 2) > config.memory_threshold:
            print(f"GPU memory usage exceeds {config.memory_threshold}MB threshold. Breaking.")
            return False

        # Training step
        inputs, labels = inputs.to(config.device), labels.to(config.device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        if config.use_l1:
            l1_norm = sum(p.abs().sum() for p in model.parameters())
            loss = loss + config.l1_lambda * l1_norm

        loss.backward()
        optimizer.step()

        # Update metrics
        epoch_train_loss += loss.item()
        num_batches += 1
        example_ct += inputs.size(0)

        # Log batch metrics
        if step + 1 < n_steps_per_epoch:
            wandb.log({
                "train/train_loss_batch": loss,
                "train/epoch_float": (step + 1 + (n_steps_per_epoch * epoch)) / n_steps_per_epoch,
                "train/example_ct": example_ct,
            })

    # Validation and metrics
    val_loss, accuracy, precision, recall = validate_model(
        model=model, 
        val_dl=val_loader,
        loss_func=criterion,
        device=config.device
    )

    # Log metrics
    metrics = {
        "train/epoch": epoch + 1,
        "train/train_loss_epoch": epoch_train_loss / num_batches,
        "val/val_loss": val_loss,
        "val/val_accuracy": accuracy,
        "val/precision": precision,
        "val/recall": recall,
        "time/epoch_minutes": (time.time() - start_time) / 60
    }
    wandb.log(metrics)
    print(f"Epoch loss: {metrics['train/train_loss_epoch']:.4f}")
    print(f"Validation loss: {metrics['val/val_loss']:.4f}")

    # Save best model
    save_callback(val_loss, model, os.path.join(output_dir, "trained_model.pth"))

    # Optionally also save this epoch's checkpoint unconditionally, so checkpoint-selection
    # strategies other than "lowest single-epoch val_loss" can be tried post-hoc without
    # needing to retrain -- SaveBestModel only keeps the single best-so-far checkpoint,
    # overwritten on every improvement, so other epochs' weights are otherwise unrecoverable.
    if config.save_all_epochs:
        epoch_dir = os.path.join(output_dir, "epoch_checkpoints")
        os.makedirs(epoch_dir, exist_ok=True)
        torch.save(
            {'model_state_dict': model.state_dict(), 'val_metric': val_loss, 'epoch': epoch},
            os.path.join(epoch_dir, f"epoch_{epoch:02d}.pth")
        )

    # Update learning rate
    if scheduler:
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        print(f"Learning rate: {current_lr}")
        wandb.log({"train/learning_rate": current_lr})

    return True

def print_config(config: TrainingConfig):
    """Print training configuration settings."""
    print("\nTraining Configuration:")
    print("----------------------")
    for field, value in vars(config).items():
        print(f"{field}: {value}")
    print("----------------------\n")

def train(config: TrainingConfig,
          build_model_fn: Optional[Callable[[TrainingConfig], torch.nn.Module]] = None,
          extra_transform=None):
    """Main training function.

    build_model_fn: if given, called as build_model_fn(config) to construct the model instead
        of the default DeepFlare_ViT. Lets other architectures (e.g. the pretrained vit_l_16 in
        train_pretrained.py) reuse this entire loop unmodified.
    extra_transform: if given, appended to both the train and val transform pipelines (before
        the random-flip augmentation, since it's cheaper to flip a smaller resized tensor) --
        e.g. v2.Resize(224) for architectures that don't take the dataset's native 512x512 input.
    """
    if config.seed is not None:
        set_seed(config.seed)

    # Initialize wandb
    wandb.init(project="flare_torch", config=vars(config), name=config.run_name)

    # Print configuration
    print_config(config)

    # Setup model
    if build_model_fn is None:
        model = DeepFlare_ViT(
            height=config.image_height,
            n_classes=config.n_classes,
            n_passbands=config.n_channels
        ).model
    else:
        model = build_model_fn(config)

    # Load statistics
    with open(config.stats_file, 'rb') as f:
        stats = pickle.load(f)
    means = [stats['mean'][f'channel_{i}'] for i in config.channel_indices]
    stds = [stats['std'][f'channel_{i}'] for i in config.channel_indices]

    # Build transform pipelines
    train_transforms = [AIALogTransform(means, stds)]
    val_transforms = [AIALogTransform(means, stds)]
    if extra_transform is not None:
        train_transforms.append(extra_transform)
        val_transforms.append(extra_transform)
    train_transforms += [v2.RandomHorizontalFlip(p=0.5), v2.RandomVerticalFlip(p=0.5)]

    # Create datasets
    train_dataset = aia_euv(
        config.json_path,
        subset='training',
        transform=v2.Compose(train_transforms),
        channel_indices=config.channel_indices
    )
    val_dataset = aia_euv(
        config.json_path,
        subset='validation',
        transform=v2.Compose(val_transforms),
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

    # Setup training
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
        if config.scheduler_type == "cosine_annealing" else None
    )

    # Create output directory
    output_dir = os.path.join("outputs", config.run_name or wandb.run.name)
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

def add_common_training_args(parser: argparse.ArgumentParser, default_memory_threshold: int = TrainingConfig.memory_threshold) -> None:
    """Add the CLI flags shared by train.py and train_pretrained.py.

    Kept in one place so the two entry points' flag sets can't silently drift apart --
    only model construction and the transform pipeline differ between them, see train().
    """
    parser.add_argument("--json-path", required=True,
                       help="Path to JSON file containing dataset information")
    parser.add_argument("--stats-file", required=True,
                       help="Path to statistics file containing means and stds")
    parser.add_argument("--batch-size", type=int, default=TrainingConfig.batch_size,
                       help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=TrainingConfig.epochs,
                       help="Number of epochs to train")
    parser.add_argument("--lr", type=float, default=TrainingConfig.learning_rate,
                       help="Learning rate")
    parser.add_argument("--scheduler", type=str, choices=["cosine_annealing", "step_lr", None],
                       help="Learning rate scheduler type")
    parser.add_argument("--retrain", action="store_true",
                       help="Resume training from checkpoint")
    parser.add_argument("--trained-model-path",
                       help="Path to pretrained model checkpoint")
    parser.add_argument("--use-l1", action="store_true",
                       help="Enable L1 regularization")
    parser.add_argument("--l1-lambda", type=float, default=TrainingConfig.l1_lambda,
                       help="L1 regularization strength")
    parser.add_argument("--channels", type=int, nargs="+", default=None,
                       help="AIA passbands to use, e.g. --channels 94 131. "
                            f"Choices: {all_wavelengths}. Default: all 7, in wavelength order.")
    parser.add_argument("--memory-threshold", type=int, default=default_memory_threshold,
                       help="Abort epoch if allocated GPU memory exceeds this many MB")
    parser.add_argument("--save-all-epochs", action="store_true",
                       help="Also save every epoch's checkpoint to <output_dir>/epoch_checkpoints/, "
                            "not just the single best-val_loss one")
    parser.add_argument("--run-name", type=str, default=None,
                       help="Deterministic name for this run/output dir, overriding wandb's random name")
    parser.add_argument("--seed", type=int, default=None,
                       help="Fix torch/numpy/python RNG state for a reproducible run "
                            "(weight init, sample order, dropout, etc.). Default: unseeded.")


def parse_args() -> TrainingConfig:
    """Parse command line arguments and create config."""
    parser = argparse.ArgumentParser()
    add_common_training_args(parser)

    args = parser.parse_args()

    # Validate that --retrain is not given without --trained-model-path
    if args.retrain and not args.trained_model_path:
        parser.error("--retrain requires --trained-model-path to be specified.")

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
        scheduler_type=args.scheduler,
        retrain=args.retrain,
        trained_model_path=args.trained_model_path,
        use_l1=args.use_l1,
        l1_lambda=args.l1_lambda,
        channel_indices=channel_indices,
        memory_threshold=args.memory_threshold,
        run_name=args.run_name,
        save_all_epochs=args.save_all_epochs,
        seed=args.seed
    )

if __name__ == "__main__":
    config = parse_args()
    train(config)
