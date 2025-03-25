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


if __name__ == "__main__":
    backend.set_image_data_format("channels_first")
    parser = argparse.ArgumentParser()
    parser.add_argument('-json-path', '--json-path', default="solar_dataset.json")
    args = parser.parse_args()

    target_ratios = {0: 0.5, 1: 0.5}  # 70% class 1, 30% class 0
    train_ds = ml_dataset(args.json_path).get_tfds(subset_name="training", do_shuffle=True,
                                                   balanced=True, target_ratios=target_ratios)

    test_ds = ml_dataset(args.json_path).get_tfds(subset_name="validation", do_shuffle=True,
                                                   balanced=True, target_ratios=target_ratios)
    train_ds = train_ds.batch(32)
    test_ds = test_ds.batch(32)

    X_train, y_train = extract_features_from_generator(train_ds)
    X_test, y_test = extract_features_from_generator(test_ds)

    # Convert the dataset into DMatrix format for XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

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

    # Train the model
    num_rounds = 100  # Number of boosting rounds
    bst = xgb.train(params, dtrain, num_rounds)

    # Make predictions on the test set
    y_pred_prob = bst.predict(dtest)

    # Evaluate the model using Binary Cross-Entropy (BCE) loss
    bce_loss = log_loss(y_test, y_pred_prob)
    print(f"Binary Cross-Entropy Loss: {bce_loss}")

    # You can also convert probabilities to binary predictions (0 or 1) using a threshold (e.g., 0.5)
    y_pred = (y_pred_prob > 0.5).astype(int)

    # Calculate accuracy or other metrics if needed
    accuracy = np.mean(y_pred == y_test)
    print(f"Accuracy: {accuracy}")
