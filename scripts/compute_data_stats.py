import os, sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
import numpy as np
import tensorflow as tf
import pickle
import argparse
from astro_utils.general import read_fits_single
from aarp_ml.dataset import aarp_dataset
from aarp_ml.model.training import ml_dataset, compute_mean_and_std

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-path")
    args = parser.parse_args()

    ds  = aarp_dataset(json_path=args.json_path)
    num_channels = 7
    train_ds = ml_dataset(ds).get_tfds(subset_name="training")
    train_ds = train_ds.batch(256)

    data, label = next(iter(train_ds))
    print("Data shape:", data.shape)

    data_mean, data_std = compute_mean_and_std(train_ds)

    stats = {
            'mean' : {f'channel_{i}': data_mean.numpy()[i] for i in range(num_channels)},
            'std' : {f'channel_{i}': data_std.numpy()[i] for i in range(num_channels)}
            }
    print(stats)

    pickle_file = "stats_E8.pkl"
    if not os.path.exists(pickle_file):
        with open(pickle_file, 'wb') as f:
            pickle.dump(stats, f)
    else:
        print(f"File {pickle_file} already exists")
        sys.exit(1)
