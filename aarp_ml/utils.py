import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import datetime
import json
from sunpy.time import TimeRange
from sunkit_instruments import goes_xrs
import time
import pandas as pd
import os
from collections import Counter
from astro_utils.flare import fetch_goes_data

def parse_json(json_path):
    with open(json_path) as f:
        data = json.load(f)
    return data

def get_aarp_ids(metadata:dict):
    """
    Get the AARP ids for each subset and label.
    Input:
    metadata: The metadata dictionary parsed from the json file.
    Output:
    A dictionary containing the AARP ids for each subset and label.
    """
    def extract_ids(subset, label):
        return [item['aarp_id'] for item in metadata.get(subset) if item['label'] == label]

    train_pos = extract_ids('training', 1)
    train_neg = extract_ids('training', 0)
    val_pos = extract_ids('validation', 1)
    val_neg = extract_ids('validation', 0)
    test_pos = extract_ids('test', 1)
    test_neg = extract_ids('test', 0)

    return {
        'train': {'pos': train_pos, 'neg': train_neg},
        'val': {'pos': val_pos, 'neg': val_neg},
        'test': {'pos': test_pos, 'neg': test_neg}
    }
    
def make_log_safe(data):
    min_pos_value = np.min(data[data > 0])
    epsilon = min_pos_value * 1e-5
    data_safe = np.where(data > 0, data, epsilon)
    return data_safe

def log_transform_flatten(data, method='naive'):
    if np.any(data < 0):
        raise ValueError("Data contains negative values")
    if method != 'naive':
        raise NotImplementedError
    else:
        return np.log(data[data>0])

def ks_test(data_0, data_1, sample_size=None):
    if sample_size:
    #sample_size =5000000
        data_0 = np.random.choice(data_0, size=sample_size)
        data_1 = np.random.choice(data_1, size=sample_size)
    ks_statistic, p_value = stats.ks_2samp(data_0, data_1)
    return ks_statistic, p_value

def plot_intensity_distribution(data, ax=None, xlabel=None, **kwargs):
    import seaborn as sns
    sns.set_theme()
    if ax is None:
        ax = plt.gca()

    data_min = data.min()
    data_max = data.max()
    intensity_hist, intensity_bins  = np.histogram(data, bins=50, range=(data_min, data_max), density=True)
    # plt.figure(figsize=(10,6))
    ax.bar(intensity_bins[:-1], intensity_hist, width=np.diff(intensity_bins), **kwargs)
    if xlabel:
        title = f"Distribution of {xlabel}"
    else:
        title = "Distribution"
    ax.set_title(title)
    ax.set_ylabel('Normalized counts')
    if xlabel:
        ax.set_xlabel(xlabel)
    ax.grid(True)
    ax.legend()
    return ax

def run_fetch_goes(start, end):
    if not isinstance(start, datetime.datetime):
        raise ValueError("Start and End should be of type datetime.datetime")
    try:
        goes_data_ts  = fetch_goes_data(start, end)
    except:
        start_with_z = start.isoformat().replace("+00:00", "Z")
        end_with_z = end.isoformat().replace("+00:00", "Z")
        goes_data_ts  = fetch_goes_data(start_with_z, end_with_z)
    return goes_data_ts

def plot_goes_with_aarp_sampling(timestamps, goes_ts_data):
    """
    timestamps: AARP timestamps
    """
    fig, ax = plt.subplots(figsize=(15,10))
    ax.set_xlim(timestamps.min(), timestamps.max())
    for ts in timestamps:
        ax.axvline(ts, color='grey', linestyle='--')
    goes_ts_data.plot(columns=['xrsb'])
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.title("GOES Timeseries with AARPS sampling")

def sizes_from_json(json_file):
    sizes = {}
    with open(json_file, "r") as file:
        data = json.load(file)
    for split in ["training", "validation", "test"]:
        label_counts = Counter(item["label"] for item in data[split])
        sizes[split] = dict(label_counts)
    return sizes

def download_GOES_events(t_start="2010-06-01", t_end="2018-12-31", dest=None):
    """
    dest: Path to save output for e.g., "data/GOES_event_list.csv"
    """
    # Grab all the data from the GOES database
    time_range = TimeRange(t_start, t_end)
    # Get only flares of class M1 or above
    st = time.time()
    listofresults = goes_xrs.get_goes_event_list(time_range, 'M1')
    print('Grabbed all the GOES data; there are', len(listofresults), 'events.')
    print(f'Time taken for download: {time.time()-st:.2f} seconds')

    df = pd.DataFrame(listofresults)
    if dest is not None:
        if not os.path.exists(dest):
            df.to_csv(dest, index=False)
        else:
            raise ValueError("File already exists. Not overwriting!")
    return df
