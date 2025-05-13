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
import argparse
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
from helpers.fits_parallel_download import download_urls_in_parallel
from aarp_ml.dataset import aarp_dataset
from aarp_ml.model.training import ml_dataset, compute_mean_and_std
from aarp_ml.data_prep import (get_clean_df, label_urls, select_urls, random_select_neg_urls, 
                               apply_shape_limits, pad_and_resize_in_parallel,
                               resample_on_shapes, extract_7h, process_table_on_disk,
                               split_urllist, remove_offlimb, pad_with_quiet,
                               read_from_disk, annotate_images, get_fov_limits, create_json)


class DatasetPaths:
    def __init__(self, parent_dir: str):
        self.parent_dir = parent_dir
        self.goes_event_list = os.path.join(parent_dir, "data/GOES_event_list.csv")
        self.aarp_full_urls = os.path.join(parent_dir, "data/aarps_full_urlist.txt")
        self.harp_to_noaa = os.path.join(parent_dir, "data/all_harps_with_noaa_ars.txt")
        self.stats_pickle = os.path.join(parent_dir, "stats_E8.pkl")
        self.json_filename = os.path.join(parent_dir, "solar_dataset_xx.json")
        self.combined_dl_list = os.path.join(parent_dir, "data/combined_dl_list.csv")
        # Directory structure
        self.pos_dir_7h = Path("/data/linn/E8/compressed/pos/")
        self.neg_dir_7h = Path("/data/linn/E8/compressed/neg/")
        self.pos_dir_single = "/data/linn/E8/extracted/pos"
        self.neg_dir_single = "/data/linn/E8/extracted/neg"

def get_download_list(goes_event_list, aarps_full_urls, harp_to_noaa, goes_class="X"):
    goes_df, aarps_clean_df = get_clean_df(goes_event_list, aarps_full_urls, harp_to_noaa)
    goes_df = goes_df[goes_df['goes_class'].apply(lambda x: x[0]) == goes_class]
    aarps_url_labelled_df = label_urls(aarps_clean_df, goes_df)
    pos_url_df = aarps_url_labelled_df[ aarps_url_labelled_df.label==1]
    #pos_url_annot_df = select_urls(pos_url_df, goes_df)
    #pos_url_selected_df = pos_url_annot_df[pos_url_annot_df.goes_matched_start.isna()]
    neg_url_df = aarps_url_labelled_df[ aarps_url_labelled_df.label==0]
    #neg_url_selected_df = random_select_neg_urls(neg_url_df, pos_url_selected_df.AARP.nunique()*imbalance_factor)
    #return pos_url_selected_df, neg_url_selected_df
    return pos_url_df, neg_url_df

def download_data(pos_urls_df: pd.DataFrame, neg_urls_df: pd.DataFrame, paths: DatasetPaths, args) -> pd.DataFrame:
    pos_urls = pos_urls_df.urls

    if args.download == True:
        download_urls_in_parallel(pos_urls, paths.pos_dir_7h, max_workers=10)

    pos_urls_df['fits_fullpath'] = pos_urls_df['urls'].apply(lambda url: os.path.join(paths.pos_dir_7h, os.path.basename(url)))
    pos_downloaded_df = pos_urls_df[pos_urls_df['fits_fullpath'].apply(os.path.exists)]

    neg_urls_selected_df = random_select_neg_urls(neg_urls_df,
                                                 pos_downloaded_df.AARP.nunique()*12)
    download_list_combined = pd.concat([pos_urls_df, neg_urls_selected_df])
    download_list_combined.to_csv(paths.combined_dl_list, index=False)

    neg_urls = neg_urls_selected_df.urls

    if args.download == True:
        download_urls_in_parallel(neg_urls, paths.neg_dir_7h, max_workers=10)

    neg_urls_selected_df['fits_fullpath'] = neg_urls_selected_df['urls'].apply(
        lambda url: os.path.join(paths.neg_dir_7h, os.path.basename(url)))
    neg_downloaded_df = neg_urls_selected_df[neg_urls_selected_df['fits_fullpath'].apply(os.path.exists)]

    downloaded_df = pd.concat([pos_downloaded_df, neg_downloaded_df])
    return downloaded_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--download', action="store_true")
    parser.add_argument('--extract', action="store_true")
    parser.add_argument('--json', action="store_true")
    parser.add_argument('--stats', action="store_true")
    args = parser.parse_args()

    st = time.time()
    paths = DatasetPaths(parent_dir)

    if args.stats and os.path.exists(paths.stats_pickle):
        raise ValueError(f"File {paths.stats_pickle} already exists")

    if args.json and os.path.exists(paths.json_filename):
        raise ValueError(f"File {paths.json_filename} already exists")

    pos_urls_df, neg_urls_df = get_download_list(
            paths.goes_event_list, paths.aarp_full_urls, paths.harp_to_noaa, goes_class="X")

    downloaded_df = download_data(pos_urls_df, neg_urls_df, paths, args)


    combined_df = process_table_on_disk(downloaded_df)
    combined_df.to_csv("combined_df.csv", index=False)

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

    selected_7h_df = grouped_df
    selected_7h_df.to_csv("data/selected_7h.csv", index=False)
    height = selected_7h_df.max_height.max()
    width = selected_7h_df.max_width.max()
    biggest_shape = (height, width)
    print(f"{biggest_shape=}")

    dim = height if height > width else width
    print(f"Using dimensions {(dim, dim)}")

    df = selected_7h_df
    print(f"{df=}")
    if args.extract == True:
        # make sure we are not extracting same file again
        assert len(pd.unique(df.fits_fullpath)) == len(df.fits_fullpath)

        print("Extracting 7h FITS observation into individual images")
        for dest, label in zip((paths.pos_dir_single, paths.neg_dir_single), (1,0)):
        # for dest, label in [(neg_dir_single, 0)]:

            files = df.fits_fullpath[df.Label==label]
            print(f"Working on samples with label:{label} first")
            print("Files to extract", len(files))
            if not os.path.exists(dest):
                os.mkdir(dest)
            print("Saving to ", dest)
            pad_and_resize_in_parallel(files, padding_func=pad_with_quiet, dest=dest, biggest_shape=(dim, dim), 
                                    targ_shape=(512, 512))
    if args.json == True:
        create_json(paths.pos_dir_single, paths.neg_dir_single, paths.json_filename)

    if args.stats == True:
        ds  = aarp_dataset(json_path=paths.json_filename)
        num_channels = 7
        with open(paths.stats_pickle, 'wb') as f:
            train_ds = ml_dataset(json_path=paths.json_filename).get_tfds(subset_name="training")
            train_ds = train_ds.batch(256)

            data_mean, data_std = compute_mean_and_std(train_ds)
            stats = {
                    'mean' : {f'channel_{i}': data_mean.numpy()[i] for i in range(num_channels)},
                    'std' : {f'channel_{i}': data_std.numpy()[i] for i in range(num_channels)}
                    }
            print(stats)

            pickle.dump(stats, f)

        print(f"Script ran for {time.time() - st} seconds")
