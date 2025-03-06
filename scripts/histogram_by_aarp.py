import os, sys
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(base_path)
import numpy as np
from tqdm import tqdm
from aarp_ml.utils import parse_json
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
"""
Script to compute histogram of intensities for a given passband and percentile level
by reading npz files in a directory containing both the AARP and attribution images.
The histogram is computed for each npz file and the counts are accumulated across all files.
The histogram data is saved to a npz file.
"""

def read_npz(file_path):
    """Read an npz file and return the label, AARP images, and attribution images."""
    try:
        data = np.load(file_path, allow_pickle=True)  # Allow pickle for flexibility
        return data['label'], data['aarp_images'], data['attbn_images']
    except Exception as e:
        print(f"Error reading file: {file_path}")
        return None  # Skip bad files

def compute_hist(data, bins):
    """Compute histogram of the input data."""
    return np.histogram(data.flatten(), bins=bins)

def process_file(file_path, channel, bin_edges, percentile_level):
    """Read data, apply masking, compute histogram."""
    result = read_npz(file_path)
    if result is None:
        return None  # Skip if file loading failed

    label, images, attbn_images = result
    passband_int = images[:, channel, :, :]
    attbn_int = attbn_images[:, channel, :, :]

    # Apply thresholding and masking
    threshold = np.percentile(attbn_int.ravel(), percentile_level)
    int_masked = np.ma.masked_where(attbn_int <= threshold, passband_int)
    int_masked = np.ma.where((int_masked < 1) & (~int_masked.mask), 1, int_masked)
    log_int = np.ma.log(int_masked)

    # Compute histogram
    counts, _ = compute_hist(log_int, bins=bin_edges)
    return (label, counts)

def main():
    json_data = parse_json("solar_dataset.json")
    passband = 94
    channel_dict = json_data['channels']
    channel = next((int(k) for k, v in channel_dict.items() if v == passband), None)

    max_int = 64000
    min_int = 1
    min_int_log = np.log(min_int)
    max_int_log = np.log(max_int)

    base_dir_path = "/data/linn/attribution_output/curious-bush-242"
    data_path = os.path.join(base_dir_path, "npz")
    file_paths = [os.path.join(data_path, file) for file in os.listdir(data_path) if file.endswith(".npz")]
    nbins = 50
    percentile_level = 99
    bin_edges = np.linspace(min_int_log, max_int_log, nbins)

    global_counts_0 = np.zeros(len(bin_edges) - 1, dtype=np.int64)
    global_counts_1 = np.zeros(len(bin_edges) - 1, dtype=np.int64)

    # Use multithreading for file loading and multiprocessing for computation
    with ThreadPoolExecutor(max_workers=8) as thread_executor:
        with ProcessPoolExecutor(max_workers=8) as process_executor:
            futures = {
                process_executor.submit(process_file, file_path, channel, bin_edges, percentile_level)
                for file_path in tqdm(file_paths)
            }
            for future in tqdm(futures):
                result = future.result()
                if result is None:
                    continue
                label, counts = result
                if label == 0:
                    global_counts_0 += counts
                elif label == 1:
                    global_counts_1 += counts
                else:
                    raise ValueError("Unknown label")

    print(f"{global_counts_0=}")
    print(f"{global_counts_1=}")

    hist_save_path = os.path.join(base_dir_path, "histogram", f"histogram_data_{passband}_pb_{percentile_level}_percentile.npz")
    np.savez(hist_save_path, global_counts_0=global_counts_0, global_counts_1=global_counts_1, bin_edges=bin_edges)

if __name__=="__main__":
    main()
