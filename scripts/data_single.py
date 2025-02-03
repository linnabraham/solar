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
import time
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
from helpers.fits_parallel_download import download_urls_in_parallel
from aarp_ml.dataset import aarp_dataset
from aarp_ml.model.training import ml_dataset, compute_mean_and_std
from aarp_ml.data_prep import (get_download_list, get_fov_limits, apply_shape_limits,
                               resample_on_shapes, extract_7h, process_table_on_disk,
                               split_urllist, remove_offlimb, pad_with_quiet,
                               read_from_disk, dir_to_json)

if __name__ == "__main__":
    st = time.time()
    goes_event_list = os.path.join(parent_dir, "./data/GOES_event_list.csv")
    aarp_full_urls = os.path.join(parent_dir, "./data/aarps_full_urlist.txt")
    harp_to_noaa = os.path.join(parent_dir, "./data/all_harps_with_noaa_ars.txt")

    pickle_file = os.path.join(parent_dir, "stats_E8.pkl")
    if os.path.exists(pickle_file):
        raise ValueError(f"File {pickle_file} already exists")

    # Create list of files to download after applying certain selections
    pos_urls_df, neg_urls_df = get_download_list(
            goes_event_list, aarp_full_urls, harp_to_noaa)

    pos_dir_7h = Path("/data/linn/E8/compressed/pos/")
    pos_dir_7h.mkdir(exist_ok=True)
    pos_urls = pos_urls_df.urls
    # download_urls_in_parallel(pos_urls, pos_dir_7h, max_workers=10)

    neg_dir_7h = Path("/data/linn/E8/compressed/neg/")
    neg_dir_7h.mkdir(exist_ok=True)
    neg_urls = neg_urls_df.urls
    # download_urls_in_parallel(neg_urls, neg_dir_7h, max_workers=10)

    table_on_disk = read_from_disk(pos_dir=pos_dir_7h, neg_dir=neg_dir_7h)
    table_on_disk[["Datetime", "AARP", "Wavelength"]] = split_urllist(table_on_disk, "fits_fullpath")[["Datetime", "AARP", "Wavelength"]]
    combined_df = process_table_on_disk(table_on_disk)
    # remove images with no location information
    combined_df.Longitude = combined_df.Longitude.replace(-999999, np.nan)
    combined_clean = combined_df[combined_df.Longitude.notna()]
    combined_clean = remove_offlimb(combined_clean)

    low_dims, high_dims = get_fov_limits(combined_clean)
    print(f"Highest in each dimenstion {high_dims}")
    shape_limited_df = apply_shape_limits(combined_clean, low_dims, high_dims)

    agg_funcs = {
        'AARP': 'first',
        'Wavelength': 'first',
        'Datetime': 'first',
        'Label': 'first',
        'img_height': 'max',
        'img_width':  'max',
        'max_abs_lon': 'first'
    }

    # Group by 'fits_fullpath' and apply aggregation
    grouped_df = shape_limited_df.groupby('fits_fullpath', as_index=False).agg(agg_funcs)
    grouped_df = grouped_df.rename({"img_height":"max_height", "img_width":"max_width"}, axis=1)

    # resample to avoid biases in intrinsic shapes between classes
    selected_7h_df = resample_on_shapes(grouped_df)

    height = selected_7h_df.max_height.max()
    width = selected_7h_df.max_width.max()
    biggest_shape = (height, width)
    print(f"{biggest_shape=}")

    dim = height if height > width else width
    print(f"Using dimensions {(dim, dim)}")

    pos_dir_single = "/data/linn/E8/extracted/pos"
    neg_dir_single = "/data/linn/E8/extracted/neg"

    extract_7h(selected_7h_df, pad_with_quiet, pos_dir_single, neg_dir_single,
               biggest_shape=(dim, dim), target_shape=(512, 512))

    json_filename = os.path.join(parent_dir, "solar_dataset_xx.json")
    dir_to_json(pos_dir_single, neg_dir_single, json_filename)

    ds  = aarp_dataset(json_path=json_filename)
    num_channels = 7
    with open(pickle_file, 'wb') as f:
        train_ds = ml_dataset(ds).get_tfds(subset_name="training")
        train_ds = train_ds.batch(256)

        data_mean, data_std = compute_mean_and_std(train_ds)
        stats = {
                'mean' : {f'channel_{i}': data_mean.numpy()[i] for i in range(num_channels)},
                'std' : {f'channel_{i}': data_std.numpy()[i] for i in range(num_channels)}
                }
        print(stats)

        pickle.dump(stats, f)

    print(f"Script ran for {time.time() - st} seconds")
