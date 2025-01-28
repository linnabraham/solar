import os
import sys
from tqdm import tqdm
import csv
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from sklearn.neighbors import KernelDensity
import numpy as np
import pandas as pd
from joblib import Memory
import pickle
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
from aarp_ml.data_prep import get_download_list
from helpers.fits_parallel_download import download_urls_in_parallel
from aarp_ml.data_prep import get_fov_limits, get_table_clean, apply_shape_limits, resample_on_shapes, extract_7h
from aarp_ml.data_prep import dir_to_json
from aarp_ml.dataset import aarp_dataset
from aarp_ml.model.training import ml_dataset, compute_mean_and_std

if __name__ == "__main__":

    goes_event_list = os.path.join(parent_dir, "./data/GOES_event_list.csv")
    aarp_full_urls = os.path.join(parent_dir, "./data/aarps_full_urlist.txt")
    harp_to_noaa = os.path.join(parent_dir, "./data/all_harps_with_noaa_ars.txt")

    # Create list of files to download after applying certain selections
    pos_urls_df, neg_urls_df = get_download_list(
            goes_event_list, aarp_full_urls, harp_to_noaa)

    pos_dir_7h = Path("/data/linn/E9/compressed/pos/")
    pos_dir_7h.mkdir(exist_ok=True)
    pos_urls = pos_urls_df.urls[:160]
    download_urls_in_parallel(pos_urls, pos_dir_7h, max_workers=10)

    neg_dir_7h = Path("/data/linn/E9/compressed/neg/")
    neg_dir_7h.mkdir(exist_ok=True)
    neg_urls = neg_urls_df.urls[:160]
    download_urls_in_parallel(neg_urls, neg_dir_7h, max_workers=10)

    pos_dir_single = "/data/linn/E9/extracted/pos"
    neg_dir_single = "/data/linn/E9/extracted/neg"

    table_7h_clean = get_table_clean(pos_dir_7h, neg_dir_7h)
    low_dims, high_dims = get_fov_limits(table_7h_clean)
    print(f"Highest in each dimenstion {high_dims}")
    shape_limited_df = apply_shape_limits(table_7h_clean, low_dims, high_dims)

    selected_7h_df = resample_on_shapes(shape_limited_df)

    height, width = selected_7h_df[["max_height", "max_width"]].iloc[np.argmax(np.prod(selected_7h_df[["max_height", "max_width"]], axis=1))]

    extract_7h(selected_7h_df, pos_dir_single, neg_dir_single,
               biggest_shape=(int(high_dims[0]),int(high_dims[1])), target_shape=(512, 512))

    json_filename = os.path.join(parent_dir, "solar_dataset_xx_test.json")
    dir_to_json(pos_dir_single, neg_dir_single, json_filename)

    ds  = aarp_dataset(json_path=json_filename)
    num_channels = 7
    train_ds = ml_dataset(ds).get_tfds(subset_name="training")
    train_ds = train_ds.batch(256)

    data_mean, data_std = compute_mean_and_std(train_ds)
    stats = {
            'mean' : {f'channel_{i}': data_mean.numpy()[i] for i in range(num_channels)},
            'std' : {f'channel_{i}': data_std.numpy()[i] for i in range(num_channels)}
            }
    print(stats)

    pickle_file = os.path.join(parent_dir, "stats_E8_test.pkl")
    if not os.path.exists(pickle_file):
        with open(pickle_file, 'wb') as f:
            pickle.dump(stats, f)
    else:
        print(f"File {pickle_file} already exists")
        sys.exit(1)
