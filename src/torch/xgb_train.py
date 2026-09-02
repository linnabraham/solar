import argparse
import os
import wandb
from tqdm import tqdm
from typing import Optional, List
from dataclasses import dataclass
import numpy as np
import torch.multiprocessing
from sklearn.metrics import log_loss
from src.torch.vit.train import print_config
from src.torch.train_alexnet import init_data
import xgboost as xgb

# Multi-worker DataLoader IPC defaults to the 'file_descriptor' sharing strategy,
# which burns one fd per tensor handed back from each worker. Over ~1000+ batches
# with num_workers>0 that exceeds the shell's default ulimit -n and kills the
# workers ("Too many open files"). 'file_system' uses shared-memory files instead,
# independent of ulimit.
torch.multiprocessing.set_sharing_strategy('file_system')

@dataclass
class TrainingConfig:
    # Required parameters
    json_path: str
    stats_file: str

    # Optional training parameters
    batch_size: int = 32
    retrain: bool = False
    trained_model_path: Optional[str] = None
    run_name: Optional[str] = None

    # Model parameters
    image_height: int = 512
    n_classes: int = 2
    n_channels: int = 7
    channel_indices: Optional[List[int]] = None  # indices into passbands; None = all 7

    # System parameters
    device: str = "cuda:0"
    memory_threshold: int = 5000  # GPU memory threshold measured in megabytes
    num_workers: int = 0  # DataLoader worker processes for parallel FITS I/O
    simple_stats: tuple = ("min", "max", "mean")
    percentiles: tuple = (60, 80, 90, 95, 98, 99)
    passbands: tuple = ('94', '131', '171', '193', '211', '304', '335')

    def __post_init__(self):
        if self.channel_indices is None:
            self.channel_indices = list(range(self.n_channels))
        else:
            self.n_channels = len(self.channel_indices)


def _batch_channel_stats(images: np.ndarray, percentiles) -> np.ndarray:
    """Vectorized [min, max, mean, percentiles..., skew, kurt] per channel, for a whole batch.

    Args:
        images: (B, C, H, W) array.
        percentiles: sequence of percentile values in [0, 100].

    Returns:
        (B, C * (3 + len(percentiles) + 2)) array, channel-major to match get_feature_names().
    """
    flat = images.reshape(images.shape[0], images.shape[1], -1)  # (B, C, N)
    mins  = flat.min(axis=-1)
    maxs  = flat.max(axis=-1)
    means = flat.mean(axis=-1)
    pcts  = np.moveaxis(np.percentile(flat, percentiles, axis=-1), 0, -1)  # (B, C, n_p)
    std   = flat.std(axis=-1, ddof=1)  # unbiased, matches the old torch.std(unbiased=True)
    skew  = ((flat - means[..., None]) ** 3).mean(axis=-1) / std ** 3
    kurt  = ((flat - means[..., None]) ** 4).mean(axis=-1) / std ** 4 - 3  # excess kurtosis

    stats = np.concatenate(
        [mins[..., None], maxs[..., None], means[..., None], pcts,
         skew[..., None], kurt[..., None]],
        axis=-1,
    )  # (B, C, n_stats_per_channel)
    return stats.reshape(stats.shape[0], -1)


def extract_stats_generator(dataset, config: TrainingConfig):
    """Extract per-image statistical features for every sample in *dataset* (a DataLoader).

    Vectorized across the batch dimension so every sample is used regardless of batch_size --
    the earlier per-image-loop version only kept images[0] of each batch, silently dropping
    the other batch_size-1 samples (~97% of the data at batch_size=32).
    """
    features = []
    labels = []
    for images, label in tqdm(dataset, desc="Extracting features"):
        features.append(_batch_channel_stats(images.numpy(), config.percentiles))
        labels.append(label.numpy())

    return np.concatenate(features, axis=0), np.concatenate(labels, axis=0)

def get_feature_names(config:TrainingConfig):
    stat_names = list(config.simple_stats)
    percentile_names = [f'p{percentile}' for percentile in config.percentiles]
    stat_names.extend(percentile_names)
    stat_names.extend(["skew","kurt"])
    feat_names = [f'{pb}_{stat_name}' for pb in config.passbands for stat_name in stat_names]
    return feat_names

def train_and_eval(config: TrainingConfig):
    """Main training function."""
    # Initialize wandb
    wandb.init(project="flare_xgb", config=vars(config), name=config.run_name)

    # Print configuration
    print_config(config)

    metadata, transform, device, train_loader, val_loader = init_data(config)

    # Create output directory
    output_dir = os.path.join("outputs", config.run_name or wandb.run.name)
    os.makedirs(output_dir, exist_ok=True)

    X_train, y_train = extract_stats_generator(train_loader, config)
    X_val, y_val = extract_stats_generator(val_loader, config)

    # Convert the dataset into DMatrix format for XGBoost
    feature_names = get_feature_names(config)
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_names)
    evals = [(dtrain, 'train'), (dval, 'validation')]  # Specify datasets for evaluation
    evals_result = {}  # Dictionary to store evaluation results

    # Set up the parameters for XGBoost
    params = {
        'objective': 'binary:logistic',  # Binary classification
        'eval_metric': 'logloss',       # Logarithmic loss (equivalent to BCE)
        'max_depth': 6,
        'eta': 0.1,                     # Learning rate
        'subsample': 0.8,               # Subsample ratio of the training instances
        'colsample_bytree': 0.8,        # Subsample ratio of columns when constructing each tree
        'seed': 42                      # Random seed for reproducibility
    }

    best_model_path = os.path.join(output_dir, "best_xgboost_model.json")  # Path to save the best model
    print(f"Saving best model to {best_model_path}")

    # Train the model
    num_rounds = 100  # Number of boosting rounds

    bst = xgb.train(
        params,
        dtrain,
        num_boost_round=num_rounds,
        evals=evals,
        evals_result=evals_result,
        early_stopping_rounds=10,
        #verbose_eval=False  # Suppress verbose output for each round
        verbose_eval=True
    )

    bst.save_model(best_model_path)  # Save the best model

    # Log the evaluation results
    print("Evaluation results:")

    for key, value in evals_result.items():
        print(f"{key}: {value}")

    # Make predictions on the val set
    y_pred_prob = bst.predict(dval)

    # Evaluate the model using Binary Cross-Entropy (BCE) loss
    bce_loss = log_loss(y_val, y_pred_prob)
    print(f"Binary Cross-Entropy Loss: {bce_loss}")

    # You can also convert probabilities to binary predictions (0 or 1) using a threshold (e.g., 0.5)
    y_pred = (y_pred_prob > 0.5).astype(int)

    # Calculate accuracy or other metrics if needed
    accuracy = np.mean(y_pred == y_val)
    print(f"Accuracy: {accuracy}")

def parse_args() -> TrainingConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-path", default="solar_dataset.json",
                       help="Path to JSON file containing dataset information")
    parser.add_argument("--stats-file", default="stats.pkl",
                       help="Path to statistics file containing means and stds")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Batch size for feature extraction")
    parser.add_argument("--num-workers", type=int, default=0,
                       help="DataLoader worker processes for parallel FITS I/O (default: 0, sequential)")
    parser.add_argument("--run-name", type=str, default=None,
                       help="Deterministic name for this run/output dir, overriding wandb's random name")
    args = parser.parse_args()

    return TrainingConfig(
        json_path=args.json_path,
        stats_file=args.stats_file,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        run_name=args.run_name,
    )

if __name__=="__main__":
    config = parse_args()
    train_and_eval(config)
