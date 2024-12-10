import os, sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
import numpy as np
import tensorflow as tf
from astro_utils.general import read_fits_single
from aarp_ml.dataset import aarp_dataset
from aarp_ml.model.training import get_tfds, compute_mean_and_std

if __name__ == "__main__":
    ds  = aarp_dataset(json_path=os.path.expanduser("~/july/solar/solar_dataset.json"))
    num_channels = 7
    train_ds = get_tfds(ds, subset_name="training")
    train_ds = train_ds.batch(64)

    data, label = next(iter(train_ds))
    print("Data shape:", data.shape)

    mean, std = compute_mean_and_std(train_ds)

    stats = {
            'mean' : {f'channel_{i}': data_mean.numpy()[i] for i in range(num_channels)},
            'std' : {f'channel_{i}': data_std.numpy()[i] for i in range(num_channels)}
            }
    print(stats)

    if not os.path.exists('stats.pkl'):
        with open('stats.pkl', 'wb') as f:
            pickle.dump(stats, f)
    else:
        print(f"File stats.pkl already exists")
        sys.exit(1)
