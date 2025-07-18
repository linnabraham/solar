import os
import time
import argparse
import math
import wandb
from tqdm import tqdm
import pickle
from torchvision.transforms import v2
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from collections import Counter
from sklearn.metrics import confusion_matrix
from aarp_ml.torch.dataset import aia_euv, AIALogTransform
from aarp_ml.torch.model import DeepFlare_ViT, SaveBestModel
from src.torch.vit.config import TrainingConfig

def get_weighted_sampler(dataset) -> WeightedRandomSampler:
    """Create a sampler that handles class imbalance."""
    # Count instances of each class
    class_counts = Counter(dataset.labels)
    total_samples = sum(class_counts.values())
    # Compute class weights (inverse of frequency)
    class_weights = {cls: total_samples / count for cls, count in class_counts.items()}
    # Assign a weight to each sample based on its class
    sample_weights = [class_weights[label] for label in dataset.labels]
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

def train(config: TrainingConfig):
    """Main training function."""
    # Initialize wandb
    wandb.init(project="flare_torch", config=vars(config))

    # Print configuration
    print_config(config)

    # Setup model
    model = DeepFlare_ViT(
        height=config.image_height,
        n_classes=config.n_classes,
        n_passbands=config.n_channels
    ).model

    # Load statistics
    with open(config.stats_file, 'rb') as f:
        stats = pickle.load(f)
    means = [stats['mean'][f'channel_{i}'] for i in range(config.n_channels)]
    stds = [stats['std'][f'channel_{i}'] for i in range(config.n_channels)]

    # Create datasets
    train_dataset = aia_euv(
        config.json_path,
        subset='training',
        transform=v2.Compose([
            AIALogTransform(means, stds),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5)
        ])
    )
    val_dataset = aia_euv(
        config.json_path,
        subset='validation',
        transform=v2.Compose([AIALogTransform(means, stds)])
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
    output_dir = os.path.join("outputs", wandb.run.name)
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
    """Parse command line arguments and create config."""
    parser = argparse.ArgumentParser()
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

    args = parser.parse_args()

    # Validate that --retrain is not given without --trained-model-path
    if args.retrain and not args.trained_model_path:
        parser.error("--retrain requires --trained-model-path to be specified.")

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
        l1_lambda=args.l1_lambda
    )

if __name__ == "__main__":
    config = parse_args()
    train(config)
