import os
import wandb
from tqdm import tqdm
import torch
from typing import Optional, List
from dataclasses import dataclass
import numpy as np
from sklearn.metrics import log_loss
from vit.scripts.train import print_config
from torch_src.train import init_data
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
    feature_type: str = "simple"
    trained_model_path: Optional[str] = None

    # Model parameters
    image_height: int = 512
    n_classes: int = 2
    n_channels: int = 7

    # System parameters
    device: str = "cuda:0"
    memory_threshold: int = 5000  # GPU memory threshold measured in megabytes

# Assuming you have a TensorFlow or Torch generator that yields (images, labels)
def extract_simple_stats_generator(dataset):
    features = []
    labels = []
    dataset_iter = iter(dataset)
    for images, label in tqdm(dataset_iter):
        # Assuming images shape is (batch_size, height, width, channels)
        images, label = images.to(config.device), label.to(config.device)
        for i in range(images.shape[0]):
            image = images[i]
            # Calculate min, max, mean for each channel
            channel_stats = []
            for channel in range(image.shape[0]):
                channel_data = image[channel, :, :]
                channel_stats.extend([torch.min(channel_data).item(), torch.max(channel_data).item(), torch.mean(channel_data).item()])

            features.append(channel_stats)
            labels.append(label[0].item())  # Assuming label is a single value for the image

    return np.array(features), np.array(labels)

def extract_percentiles_generator(dataset):
    features = []
    labels = []
    dataset_iter = iter(dataset)
    for images, label in dataset_iter:
        # Assuming images shape is (batch_size, height, width, channels)
        # We take the first image in the batch
        image = images[0].numpy()

        # Calculate fixed percentile values for each channel
        channel_stats = []
        for channel in range(image.shape[0]):
            channel_data = image[channel, :, :]
            channel_stats.extend([np.percentile(channel_data,60), np.percentile(channel_data,80), np.percentile(channel_data,90),
                                  np.percentile(channel_data,95), np.percentile(channel_data,98), np.percentile(channel_data,99)])

        features.append(channel_stats)
        labels.append(label[0])  # Assuming label is a single value for the image

    return np.array(features), np.array(labels)

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

    if config.feature_type == "simple":
        X_train, y_train = extract_simple_stats_generator(train_loader)
        X_val, y_val = extract_simple_stats_generator(val_loader)
    elif config.feature_type == "percentile":
        X_train, y_train = extract_percentiles_generator(train_loader)
        X_val, y_val = extract_percentiles_generator(val_loader)
    else:
        raise ValueError("Method not implemented")

    # Convert the dataset into DMatrix format for XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
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
