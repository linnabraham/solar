import os, sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from tqdm import tqdm
from aarp_ml.model.training import ml_dataset

# Assuming you have a TensorFlow generator that yields (images, labels)
def extract_features_from_generator(dataset):
    features = []
    labels = []
    dataset_iter = iter(dataset)
    for images, label in tqdm(dataset_iter):
        # Assuming images shape is (batch_size, height, width, channels)
        # We take the first image in the batch
        image = images[0]

        # Calculate min, max, mean for each channel
        channel_stats = []
        for channel in range(image.shape[0]):
            channel_data = image[channel, :, :]
            channel_stats.extend([np.min(channel_data), np.max(channel_data), np.mean(channel_data)])

        features.append(channel_stats)
        labels.append(label[0])  # Assuming label is a single value for the image

    return np.array(features), np.array(labels)

def extract_percentiles_generator(dataset):
    features = []
    labels = []
    dataset_iter = iter(dataset)
    for images, label in tqdm(dataset_iter):
        # Assuming images shape is (batch_size, height, width, channels)
        # We take the first image in the batch
        image = images[0]

        # Calculate fixed percentile values for each channel
        channel_stats = []
        for channel in range(image.shape[0]):
            channel_data = image[channel, :, :]
            channel_stats.extend([np.percentile(channel_data,60), np.percentile(channel_data,80), np.percentile(channel_data,90),
                                  np.percentile(channel_data,95), np.percentile(channel_data,98), np.percentile(channel_data,99)])

        features.append(channel_stats)
        labels.append(label[0])  # Assuming label is a single value for the image

    return np.array(features), np.array(labels)


if __name__ == "__main__":
    backend.set_image_data_format("channels_first")
    parser = argparse.ArgumentParser()
    parser.add_argument('-json-path', '--json-path', default="solar_dataset.json")
    args = parser.parse_args()

    target_ratios = {0: 0.5, 1: 0.5}  # 70% class 1, 30% class 0
    train_ds = ml_dataset(args.json_path).get_tfds(subset_name="training", do_shuffle=True,
                                                   balanced=True, target_ratios=target_ratios)

    val_ds = ml_dataset(args.json_path).get_tfds(subset_name="validation", do_shuffle=True,
                                                   balanced=True, target_ratios=target_ratios)
    train_ds = train_ds.batch(32)
    val_ds = val_ds.batch(32)

    X_train, y_train = extract_features_from_generator(train_ds)
    X_val, y_val = extract_features_from_generator(val_ds)

    # Convert the dataset into DMatrix format for XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    evals = [(dtrain, 'train'), (dval, 'validation')]  # Specify datasets for evaluation
    evals_result = {}  # Dictionary to store evaluation results

    # Set up the parameters for XGBoost
    params = {
        'objective': 'binary:logistic',  # Binary classification
        'eval_metric': 'logloss',       # Logarithmic loss (equivalent to BCE)
        'max_depth': 6,                 # Maximum depth of a tree
        'eta': 0.1,                     # Learning rate
        'subsample': 0.8,               # Subsample ratio of the training instances
        'colsample_bytree': 0.8,        # Subsample ratio of columns when constructing each tree
        'seed': 42                      # Random seed for reproducibility
    }
    best_model_path = "best_xgboost_model.json"  # Path to save the best model
    num_rounds = 100  # Number of boosting rounds
    bst = xgb.train(
        params,
        dtrain,
        num_boost_round=num_rounds,
        evals=evals,
        evals_result=evals_result,
        early_stopping_rounds=10,
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
