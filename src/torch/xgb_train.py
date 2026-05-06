import os
import wandb
from tqdm import tqdm
import torch
from typing import Optional, List
from dataclasses import dataclass
import numpy as np
from sklearn.metrics import log_loss
from src.torch.vit.train import print_config
from src.torch.train_alexnet import init_data
import xgboost as xgb
import numpy as np
from sklearn.metrics import log_loss
from tqdm import tqdm

@dataclass
class TrainingConfig:
    # Required parameters
    json_path: str
    stats_file: str

    # Optional training parameters
    batch_size: int = 32
    retrain: bool = False
    trained_model_path: Optional[str] = None

    # Model parameters
    image_height: int = 512
    n_classes: int = 2
    n_channels: int = 7

    # System parameters
    device: str = "cuda:0"
    memory_threshold: int = 5000  # GPU memory threshold measured in megabytes
    simple_stats: tuple = ("min", "max", "mean")
    percentiles: tuple = (60, 80, 90, 95, 98, 99)
    passbands: tuple = ('94', '131', '171', '193', '211', '304', '335')

# Assuming you have a TensorFlow or Torch generator that yields (images, labels)
def extract_stats_generator(dataset, config:TrainingConfig):
    features = []
    labels = []
    dataset_iter = iter(dataset)
    for images, label in dataset_iter:
        # Assuming images shape is (batch_size, height, width, channels)
        # We take the first image in the batch
        image = images[0].numpy()

        # Calculate basic stats and fixed percentile values for each channel
        channel_stats = []

        for channel in range(image.shape[0]):
            channel_data = image[channel, :, :]
            percentile_feats = [np.percentile(channel_data,percentile) for percentile in config.percentiles]
            channel_data = torch.from_numpy(channel_data)
            channel_stats.extend([torch.min(channel_data).item(), torch.max(channel_data).item(), torch.mean(channel_data).item()])
            channel_stats.extend(percentile_feats)
            channel_stats.extend([torch_skew(channel_data).item(), torch_kurtosis(channel_data).item()])

        features.append(channel_stats)
        labels.append(label[0])  # Assuming label is a single value for the image

    return np.array(features), np.array(labels)

def get_feature_names(config:TrainingConfig):
    stat_names = list(config.simple_stats)
    percentile_names = [f'p{percentile}' for percentile in config.percentiles]
    stat_names.extend(percentile_names)
    stat_names.extend(["skew","kurt"])
    feat_names = [f'{pb}_{stat_name}' for pb in config.passbands for stat_name in stat_names]
    return feat_names

def torch_skew(x, dim=None, unbiased=True):
    mean = x.mean(dim, keepdim=True)
    std = x.std(dim, unbiased=unbiased, keepdim=True)
    skew = ((x - mean) ** 3).mean(dim) / (std.squeeze() ** 3)
    return skew

def torch_kurtosis(x, dim=None, unbiased=True, excess=True):
    mean = x.mean(dim, keepdim=True)
    std = x.std(dim, unbiased=unbiased, keepdim=True)
    kurt = ((x - mean) ** 4).mean(dim) / (std.squeeze() ** 4)
    if excess:
        kurt -= 3
    return kurt

def train_and_eval(config: TrainingConfig):
    """Main training function."""
    # Initialize wandb
    wandb.init(project="flare_xgb", config=vars(config))

    # Print configuration
    print_config(config)

    metadata, transform, device, train_loader, val_loader = init_data(config)

    # Create output directory
    output_dir = os.path.join("output", wandb.run.name)
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
        'max_depth': 1000,                 # Maximum depth of a tree
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

if __name__=="__main__":
    config = TrainingConfig(json_path="solar_dataset.json", stats_file="stats.pkl")
    train_and_eval(config)
