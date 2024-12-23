import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from astro_utils.flare import fetch_goes_data
import datetime

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
