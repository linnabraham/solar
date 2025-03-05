import os, sys
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(base_path)
import numpy as np
from tqdm import tqdm
from aarp_ml.utils import parse_json
"""
Script to compute histogram of intensities for a given passband and percentile level
by reading npz files in a directory containing both the AARP and attribution images.
The histogram is computed for each npz file and the counts are accumulated across all files.
The histogram data is saved to a npz file.
"""

def read_npz(file_path):
    try:
        data = np.load(file_path)
    except Exception as e:
        print(f"Error reading file:{file_path}")
        raise e
    aarp_images = data['aarp_images']
    attbn_images = data['attbn_images']
    label = data['label']
    return label, aarp_images, attbn_images

def compute_hist(data, bins):
    counts, bin_edges = np.histogram(data.flatten(), bins=bins)
    return counts, bin_edges

def main():
    json_data = parse_json("solar_dataset.json")
    passband = 94
    channel_dict = json_data['channels']
    channel = next((int(k) for k,v in channel_dict.items() if v == passband), None)

    max_int = 64000
    min_int = 1
    min_int_log = np.log(min_int)
    max_int_log = np.log(max_int)

    dir_path = "/data/linn/attribution_output/electric-star-195"
    file_paths = [os.path.join(dir_path, file) for file in os.listdir(dir_path) if file.endswith(".npz")]
    nbins = 50
    percentile_level = 99
    bin_edges = np.linspace(min_int_log, max_int_log, nbins)

    global_counts_0 = np.zeros(len(bin_edges) - 1, dtype=np.int64)
    global_counts_1 = np.zeros(len(bin_edges) - 1, dtype=np.int64)

    for file_path in tqdm(file_paths):
        label, images, attbn_images = read_npz(file_path)
        passband_int = images[:,channel,:,:]
        attbn_int = attbn_images[:,channel,:,:]
        threshold = np.percentile(attbn_int.ravel(), percentile_level)
        int_masked = np.ma.masked_where(attbn_int <= threshold, passband_int)
        # Find values where the mask is False (valid values) and are < 1
        int_masked = np.ma.where((int_masked < 1) & (~int_masked.mask), 1, int_masked)
        log_int = np.ma.log(int_masked)
        #print(f"{label=}, {images.shape=}")
        counts, bin_edges = compute_hist(log_int, bins=bin_edges)
        #print(f"{counts[0]=}, {bin_edges[0],bin_edges[-1]}")
        if label == 0:
            global_counts_0 += counts
        elif label == 1:
            global_counts_1 += counts
        else:
            raise ValueError("Unknown label")
    print(f"{global_counts_0=}")
    print(f"{global_counts_1=}")
    np.savez(f"histogram_data_{passband}pb_{percentile_level}_percentile.npz", global_counts_0=global_counts_0, global_counts_1=global_counts_1, bin_edges=bin_edges)

if __name__=="__main__":
    main()
