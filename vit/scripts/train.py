import sys, os
sys.path.append(os.path.expanduser("~/july/solar/"))
from aarp_ml.torch.dataset import aia_euv, CustomTransform
from aarp_ml.torch.model import DeepFlare_ViT, SaveBestModel
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
import torch.optim.lr_scheduler

threshold = 5000  # GPU memory threshold measured in megabytes

def get_weighted_sampler(dataset):
    """
    Create a WeightedRandomSampler for handling class imbalance.

    Args:
        dataset: A dataset object with a `labels` attribute containing class labels.

    Returns:
        WeightedRandomSampler: A sampler that samples based on class distribution.
    """
    # Count occurrences of each class in the dataset
    class_counts = Counter(dataset.labels)
    total_samples = sum(class_counts.values())

    # Compute class weights (inverse of frequency)
    class_weights = {cls: total_samples / count for cls, count in class_counts.items()}

    # Assign a weight to each sample based on its class
    sample_weights = [class_weights[label] for label in dataset.labels]

    return WeightedRandomSampler(weights=sample_weights, num_samples=len(dataset), replacement=True)

def validate_model(model, val_dl, loss_func, device):
    model.eval()
    val_loss = 0.
    y_true, y_pred = [], []

    with torch.inference_mode():

        for i, (images, labels) in tqdm(enumerate(val_dl), total=len(val_dl), leave=False):
            images, labels = images.to(device), labels.to(device)

            # Forward pass ➡
            outputs = model(images)
            val_loss += loss_func(outputs, labels)*labels.size(0)
            # Compute accuracy and accumulate
            pred_scores, predicted = torch.max(outputs.data, 1)
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

        TN, FP, FN, TP = confusion_matrix(y_true, y_pred).ravel()
        precision = TP/(TP+FP) if (TP+FP) > 0 else 0
        recall = TP/(TP+FN) if (TP+FN) > 0 else 0
        print("True Positives:", TP)
        print("False Positives:", FP)
        print("True Negatives:", TN)
        print("False Negatives:", FN)

    return val_loss / len(val_dl.dataset), (TP+TN) / len(val_dl.dataset), precision, recall

def training_step(inputs, labels, model, criterion, optimizer):
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    return loss

def train_loop(train_loader, val_loader, model, device, output_dir, args):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Initialize scheduler based on the argument
    if args.scheduler == "cosine_annealing":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0)
    else:
        scheduler = None  # No scheduler

    print(f"Length of training data", len(train_loader.dataset))
    n_steps_per_epoch = math.ceil(len(train_loader.dataset) / args.batch_size)
    print(f"Steps per epoch:{n_steps_per_epoch}")

    save_best_model_callback = SaveBestModel(monitor='val_loss', mode='min')

    start_epoch = 0

    if args.retrain:
        if args.trained_model_path:
            print(f"Loading saved model from {args.trained_model_path}")
            checkpoint = torch.load(args.trained_model_path, map_location=device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'epoch' in checkpoint:
                start_epoch = checkpoint['epoch'] + 1
            else:
                start_epoch = 0  # Default start epoch
        else:
            raise FileNotFoundError(f"Checkpoint file not found at {args.trained_model_path}")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        running_loss = 0.0

        start_time = time.time()

        print(f"Epoch:{epoch}")

        for step, (inputs, labels) in tqdm(enumerate(train_loader), total=len(train_loader), leave=False):
            current_memory = torch.cuda.memory_allocated() / (1024 ** 2)
            if current_memory > threshold:
                print(f"GPU memory usage ({current_memory} MB) exceeds threshold. Breaking the script.")
                return

            inputs, labels = inputs.to(device), labels.to(device)
            loss = training_step(inputs, labels, model, criterion, optimizer)
            running_loss += loss.item() * inputs.size(0)
            metrics = {"train/train_loss": loss,
                       "train/epoch": (step + 1 + (n_steps_per_epoch * epoch)) / n_steps_per_epoch
                       }
            if step + 1 < n_steps_per_epoch:
                # Log train metrics to wandb
                wandb.log(metrics)

        # Validation step
        val_loss, accuracy, precision, recall = validate_model(model, val_loader, criterion, device)

        val_metrics = {"val/val_loss": val_loss,
                       "val/val_accuracy": accuracy,
                       "val/precision": precision,
                       "val/recall": recall}

        wandb.log({**metrics, **val_metrics})

        save_best_model_callback(val_loss, model, os.path.join(output_dir, "trained_model.pth"))

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch loss: {epoch_loss}")

        epoch_time = time.time() - start_time
        print(f"Time taken to run single epoch: {epoch_time / 60} mins")

        max_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # Convert to megabytes
        print(f"Maximum GPU memory usage: {max_memory} MB")

        max_memory_reserved = torch.cuda.max_memory_reserved() / (1024 ** 2)
        print(f"Maximum GPU memory reserved: {max_memory_reserved}")

        # Step the scheduler if it exists
        if scheduler:
            scheduler.step()

            # Log the current learning rate
            current_lr = scheduler.get_last_lr()[0]
            print(f"Learning rate for epoch {epoch}: {current_lr}")
            wandb.log({"train/learning_rate": current_lr})


def train(args):
    args_dict = vars(args)
    project_name = "flare_torch"
    wandb.init(
        project=project_name,
        config=args_dict
    )

    vit_model = DeepFlare_ViT(height=512, n_classes=2, n_passbands=7)
    model = vit_model.model

    json_path = args.json_path
    stats_file = args.stats_file

    with open(stats_file, 'rb') as f:
        stats = pickle.load(f)

    means = [stats['mean'][f'channel_{i}'] for i in range(7)]
    stds = [stats['std'][f'channel_{i}'] for i in range(7)]

    train_dataset = aia_euv(json_path, subset='training', transform=v2.Compose([
        CustomTransform(means, stds),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomVerticalFlip(p=0.5)
    ]))

    validation_dataset = aia_euv(json_path, subset='validation', transform=v2.Compose([CustomTransform(means, stds)]))

    print("Checking data specifications")
    for i in range(len(train_dataset)):
        features, label = train_dataset[i]
        print(features.shape)
        break

    print("train size", len(train_dataset))
    print("validation size", len(validation_dataset))

    # Create DataLoader with oversampling
    train_loader = DataLoader(train_dataset, batch_size=32, sampler=get_weighted_sampler(train_dataset))
    val_loader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}")

    torch.cuda.reset_peak_memory_stats()

    model.to(device)

    wandb_dir = wandb.run.name
    output_dir = os.path.join("output", wandb_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    train_loop(train_loader, val_loader, model, device, output_dir, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-json-path", "--json-path")
    parser.add_argument("-stats-file", "--stats-file")
    parser.add_argument("-batch-size", "--batch-size", type=int, default=32)
    parser.add_argument("-epochs", "--epochs", type=int, default=5)
    parser.add_argument('-lr', '--lr', type=float, default=0.001)
    parser.add_argument('--scheduler', type=str, default=None, choices=["cosine_annealing", "step_lr", None],
                        help="Learning rate scheduler to use (e.g., cosine_annealing, step_lr)")
    parser.add_argument('--retrain', action="store_true", help="Flag to resume training from a previous epoch")
    parser.add_argument('--trained-model-path', help="Location of saved model")

    args = parser.parse_args()
    print(args)
    train(args)