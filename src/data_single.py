import os
import sys
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import pandas as pd
import pickle
import time
import argparse
from datetime import datetime, timedelta
from src.fits_parallel_download import download_urls_in_parallel
from aarp_ml.dataset import aarp_dataset
from aarp_ml.model.training import ml_dataset, compute_mean_and_std
from aarp_ml.data_prep import (get_clean_df, label_urls, random_select_neg_urls, apply_shape_limits,
                               pad_and_resize_in_parallel, process_table_on_disk, remove_offlimb,
                               pad_with_quiet, get_fov_limits, create_json)

class DatasetPaths:
    def __init__(self, parent_dir: str):
        self.parent_dir = parent_dir
        self.goes_event_list = os.path.join(parent_dir, "data/GOES_event_list.csv")
        self.aarp_full_urls = os.path.join(parent_dir, "data/aarps_full_urlist.txt")
        self.harp_to_noaa = os.path.join(parent_dir, "data/all_harps_with_noaa_ars.txt")
        self.stats_pickle = os.path.join(parent_dir, "stats.pkl")
        self.json_filename = os.path.join(parent_dir, "solar_dataset.json")
        self.goes_event_with_aarp = os.path.join(parent_dir, "data/goes_df_aarp_id.csv")
        self.pos_urls_selected_downloaded = os.path.join(parent_dir, "data/pos_urls_selected_downloaded.csv")
        self.neg_urls_selected_downloaded = os.path.join(parent_dir, "data/neg_urls_selected_downloaded.csv")
        self.combined_dl_list = os.path.join(parent_dir, "data/combined_dl_list.csv")
        self.combined_processed = os.path.join(parent_dir, "data/combined_processed_df.csv")
        self.combined_clean = os.path.join(parent_dir, "data/combined_clean.csv")
        self.shape_limited = os.path.join(parent_dir, "data/shape_limited.csv")
        self.grouped_df = os.path.join(parent_dir, "data/grouped_df.csv")
        # Directory structure
        self.pos_dir_7h = Path("/data/linn/E8/compressed/pos/")
        self.neg_dir_7h = Path("/data/linn/E8/compressed/neg/")
        self.pos_dir_single = Path("/data/linn/E8/extracted/pos")
        self.neg_dir_single = Path("/data/linn/E8/extracted/neg")


def get_download_list(goes_event_list, aarps_full_urls, harp_to_noaa, goes_class="X"):
    goes_df, aarps_clean_df = get_clean_df(goes_event_list, aarps_full_urls, harp_to_noaa)
    goes_df = goes_df[goes_df['goes_class'].apply(lambda x: x[0]) == goes_class]
    aarps_url_labelled_df = label_urls(aarps_clean_df, goes_df)
    pos_url_df = aarps_url_labelled_df[ aarps_url_labelled_df.label==1]
    neg_url_df = aarps_url_labelled_df[ aarps_url_labelled_df.label==0]
    return goes_df, (pos_url_df, neg_url_df)

def download_data(pos_urls_df: pd.DataFrame, neg_urls_df: pd.DataFrame, paths: DatasetPaths, imbalance_factor=12, args=None) -> pd.DataFrame:
    pos_urls_selected_df = select_pos_urls(pos_urls_df, goes_df)

    pos_urls_selected_df['compressed_fits_fullpath'] = pos_urls_selected_df.urls.apply(
        lambda url: os.path.join(paths.pos_dir_7h, os.path.basename(url)))


    neg_urls_selected_df = random_select_neg_urls(neg_urls_df,
                                                 pos_urls_selected_df.AARP.nunique()* imbalance_factor)
    neg_urls_selected_df['compressed_fits_fullpath'] = neg_urls_selected_df['urls'].apply(
        lambda url: os.path.join(paths.neg_dir_7h, os.path.basename(url)))
    downloaded_list_combined_df = pd.concat([pos_urls_selected_df, neg_urls_selected_df])

    # save files to disk
    pos_urls_selected_df.to_csv(paths.pos_urls_selected_downloaded, index=False)
    neg_urls_selected_df.to_csv(paths.neg_urls_selected_downloaded, index=False)
    downloaded_list_combined_df.to_csv(paths.combined_dl_list, index=False)

    if args.download == True:
        pos_urls = pos_urls_selected_df.urls
        neg_urls = neg_urls_selected_df.urls
        download_urls_in_parallel(pos_urls, paths.pos_dir_7h, max_workers=10)
        download_urls_in_parallel(neg_urls, paths.neg_dir_7h, max_workers=10)

    return downloaded_list_combined_df


def check_dataset_paths(paths: DatasetPaths) -> None:
    """Check existence of all files and directories in DatasetPaths."""
    print("\nChecking dataset files and directories:")
    print("----------------------------------------")

    # Check input files
    input_files = {
        "GOES event list": paths.goes_event_list,
        "AARP URLs list": paths.aarp_full_urls,
        "HARP to NOAA mapping": paths.harp_to_noaa
    }

    print("\nInput Files:")
    for name, path in input_files.items():
        status = "✓ Found" if os.path.exists(path) else "✗ Missing"
        print(f"{name:.<25}, {path:.<25} {status}")

    # Check generated files
    generated_files = {
        "Statistics pickle": paths.stats_pickle,
        "Dataset JSON": paths.json_filename,
        "GOES events with AARP": paths.goes_event_with_aarp,
        "Positive URLs": paths.pos_urls_selected_downloaded,
        "Negative URLs": paths.neg_urls_selected_downloaded,
        "Combined download list": paths.combined_dl_list,
        "Shape limited data": paths.shape_limited,
    }

    print("\nGenerated Files:")
    for name, path in generated_files.items():
        status = "✓ Found" if os.path.exists(path) else "✗ Missing"
        print(f"{name:.<25}, {path} {status}")

    # Check directories
    directories = {
        "Positive 7h directory": paths.pos_dir_7h,
        "Negative 7h directory": paths.neg_dir_7h,
        "Positive single directory": paths.pos_dir_single,
        "Negative single directory": paths.neg_dir_single
    }

    print("\nDirectories:")
    for name, path in directories.items():
        if os.path.exists(path):
            n_files = len(list(Path(path).glob('*')))
            status = f"✓ Found ({n_files} files)"
        else:
            status = "✗ Missing"
        print(f"{name:.<25}, {path} {status}")

    print("\n----------------------------------------")
    if args.stats and os.path.exists(paths.stats_pickle):
        raise ValueError(f"File {paths.stats_pickle} already exists")

    if args.json and os.path.exists(paths.json_filename):
        raise ValueError(f"File {paths.json_filename} already exists")


def annotate_pos_urls(urldf, goes_df):
    """
    Annotate observations if they are taken after the occurence of a flare
    """
    urldf_copy = urldf.copy()
    urldf_copy['goes_matched_start'] = None
    for index, row in tqdm(urldf.iterrows(), total=len(urldf)):
        aarp_id = row['AARP']
        obs_start = datetime.strptime(row['Datetime'], "%Y.%m.%d_%H:%M:%S")
        matching_rows = goes_df[ (goes_df['harpnum'] == aarp_id) & (obs_start + timedelta(hours=6) > goes_df['start_time'][goes_df['harpnum'] == aarp_id])]
        if any(matching_rows):
            if not matching_rows.empty:
                matched_start_time = matching_rows['start_time'].values[0]
                urldf_copy.at[index, 'goes_matched_start'] = matched_start_time
    return urldf_copy

def select_pos_urls(urldf, goes_df):
    pos_url_annot_df = annotate_pos_urls(urldf, goes_df)
    pos_url_selected_df = pos_url_annot_df[pos_url_annot_df.goes_matched_start.isna()]
    return pos_url_selected_df

def select_images(combined_processed_df_org):
    """
    Remove images without timestamp information and longitude information
    Remove observations taken at the limb
    #TODO: Check if this works as intended
    Select images with a similar shape distribution as the positive sample 
    """
    combined_processed_df_org.rename({'fits_fullpath':'compressed_fits_fullpath'}, axis=1, inplace=True)

    combined_processed_df = combined_processed_df_org[combined_processed_df_org.img_time.notna()].copy()
    combined_processed_df['fits_path'] = combined_processed_df.apply(
                lambda row: f"{row['AARP']}_{row['Wavelength']}_{row['Datetime']}_TAI_{row['img_time']}.fits",
                    axis=1
                    )
    combined_processed_df['fits_fullpath'] = combined_processed_df.apply(
    lambda row: os.path.join(paths.pos_dir_single if row['Label'] == 1 else paths.neg_dir_single, row['fits_path']),
        axis=1
        )
    # remove images with no location information
    combined_processed_df.Longitude = combined_processed_df.Longitude.replace(-999999, np.nan)
    combined_clean = combined_processed_df[combined_processed_df.Longitude.notna()]
    combined_clean = remove_offlimb(combined_clean)

    low_dims, high_dims = get_fov_limits(combined_clean)
    print(f"Highest in each dimension {high_dims}")
    shape_limited_df = apply_shape_limits(combined_clean, low_dims, high_dims)
    return combined_clean, shape_limited_df

def create_grouped_df(shape_limited_df):
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
    grouped_df = shape_limited_df.groupby('compressed_fits_fullpath', as_index=False).agg(agg_funcs)
    grouped_df = grouped_df.rename({"img_height":"max_height", "img_width":"max_width"}, axis=1)
    return grouped_df

def extract_df(df, paths, dim):
    """
    Function that extracts individual images from compressed FITS files with padding applied,
    Takes as input dataframe with file paths and labels and uses separate folders based on label
    """

    # make sure we are not extracting same file again
    assert len(pd.unique(df.compressed_fits_fullpath)) == len(df.compressed_fits_fullpath)

    print("Extracting 7h FITS observation into individual images")
    for dest, label in zip((paths.pos_dir_single, paths.neg_dir_single), (1,0)):

        files = df.compressed_fits_fullpath[df.Label==label]
        print(f"Working on samples with label:{label} first")
        print("Files to extract", len(files))
        if not os.path.exists(dest):
            os.mkdir(dest)
        print("Saving to ", dest)
        pad_and_resize_in_parallel(files, padding_func=pad_with_quiet, dest=dest, biggest_shape=(dim, dim), 
                                targ_shape=(512, 512))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--download', action="store_true")
    parser.add_argument('--process', action="store_true")
    parser.add_argument('--select', action="store_true")
    parser.add_argument('--extract', action="store_true")
    parser.add_argument('--json', action="store_true")
    parser.add_argument('--stats', action="store_true")
    args = parser.parse_args()

    st = time.time()
    paths = DatasetPaths(parent_dir=os.getcwd())
    check_dataset_paths(paths)

    if args.process:
        goes_df, (pos_urls_df, neg_urls_df) = get_download_list(
                paths.goes_event_list, paths.aarp_full_urls, paths.harp_to_noaa, goes_class="X")
        goes_df.to_csv(paths.goes_event_with_aarp, index=False)

        downloaded_list_combined_df = download_data(pos_urls_df, neg_urls_df, paths, imbalance_factor=12, args=args)
        downloaded_list_combined_df.to_csv(paths.combined_dl_list, index=False)
        combined_processed_df = process_table_on_disk(downloaded_list_combined_df)
        combined_processed_df.to_csv(paths.combined_processed, index=False)

    if args.select:
        combined_processed_df_org = pd.read_csv(paths.combined_processed)
        combined_clean, shape_limited_df = select_images(combined_processed_df_org)
        grouped_df = create_grouped_df(shape_limited_df)

        # harcode problematic files that do not get actually extracted to disk and hence 
        # breaks the pipeline unless removed
        grouped_df.drop(grouped_df.query('compressed_fits_fullpath.str.startswith("/data/linn/E8/compressed/neg/2011.12.09_15:48:00_7h@1h_AARP1126")').index, inplace=True)
        shape_limited_df.drop(shape_limited_df.query('compressed_fits_fullpath.str.startswith("/data/linn/E8/compressed/neg/2011.12.09_15:48:00_7h@1h_AARP1126")').index, inplace=True)

        # save intermediate files to disk
        combined_clean.to_csv(paths.combined_clean, index=False)
        shape_limited_df.to_csv(paths.shape_limited, index=False)
        grouped_df.to_csv(paths.grouped_df, index=False)

    if args.extract == True:
        grouped_df = pd.read_csv(paths.grouped_df)
        height = grouped_df.max_height.max()
        width = grouped_df.max_width.max()
        biggest_shape = (height, width)
        print(f"{biggest_shape=}")
        dim = height if height > width else width
        print(f"Using dimensions {(dim, dim)}")
        print(f"{grouped_df=}")

        extract_df(grouped_df, paths, dim)

    if args.json == True:
        print(f"Creating json file...")
        shape_limited_df = pd.read_csv(paths.shape_limited)
        create_json(paths.pos_dir_single, paths.neg_dir_single, shape_limited_df, paths.json_filename)

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
