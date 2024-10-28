import tracemalloc
import glob
import re
from astropy.io import fits
import dask.array as da
from dask import delayed
import dask
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from concurrent.futures import ThreadPoolExecutor
from matplotlib.offsetbox import AnchoredText
import seaborn as sns
sns.set_theme()

def read_numpy(file_path):
    return np.load(file_path, mmap_mode='r')

def ks_test(data_0, data_1):
    sample_size =5000000
    data_0 = np.random.choice(data_0, size=sample_size)
    data_1 = np.random.choice(data_1, size=sample_size)
    ks_statistic, p_value = stats.ks_2samp(data_0, data_1)
    return ks_statistic, p_value

def make_dist_plots():
    # Plot and save the histogram
    # plt.figure(figsize=(10, 6))
    concatenated_arr_1, intensity_bins, histogram = process_class_npy(npy_dir_path_1)
    # plt.bar(intensity_bins[:-1], histogram, width=np.diff(intensity_bins), edgecolor='black', label="Flared")
    concatenated_arr_0, intensity_bins, histogram = process_class_npy(npy_dir_path_0)
    # plt.bar(intensity_bins[:-1], histogram, width=np.diff(intensity_bins), edgecolor='black', alpha=0.6, label="Not flared")

    #ks_statistic, p_value = stats.ks_2samp(concatenated_arr_1, concatenated_arr_0)
    data_1 = concatenated_arr_1.compute()
    data_0 = concatenated_arr_0.compute()
    sample_size =5000000
    data_1 = np.random.choice(data_1, size=sample_size)
    data_0 = np.random.choice(data_0, size=sample_size)
    # ks_statistic, p_value = stats.ks_2samp(data_1, data_0)
    # print("KS test", ks_statistic, p_value)
    # plt.title('Intensity Distribution Histogram')
    # plt.xlabel('Intensity')
    # plt.ylabel('Frequency')
    # plt.grid(True)
    # plt.legend()
    # plt.savefig("intensity_histogram_bothclass.png", dpi=300)

def find_percentiles(concatenated_arr):
    data_99p = da.percentile(concatenated_arr.flatten(), 99).compute()
    data_90p = da.percentile(concatenated_arr.flatten(), 90).compute()
    data_80p = da.percentile(concatenated_arr.flatten(), 80).compute()
    data_60p = da.percentile(concatenated_arr.flatten(), 60).compute()
    return (data_60p, data_80p, data_90p, data_99p)

def concat_data(dir_path, channel=None):

    file_names = glob.glob(dir_path+"*.npy")

    with ThreadPoolExecutor() as executor:
        dask_arrs = list(executor.map(lambda file_: da.from_array(read_numpy(file_), chunks='auto'), file_names ))

    dask_arrs = [ da.from_array(read_numpy(file_), chunks='auto') for file_ in file_names ]
    concatenated_arr = da.concatenate(dask_arrs, axis=0)
    if channel is not None:
        concatenated_arr = concatenated_arr[:,channel,:,:]
    return concatenated_arr

def concat_att(dir_path, channel=None):

    file_names = glob.glob(dir_path+"*.npy")

    with ThreadPoolExecutor() as executor:
        dask_arrs = list(executor.map(lambda file_: da.from_array(read_numpy(file_), chunks='auto'), file_names ))
    # dask_arrs = [ da.from_array(read_numpy(file_), chunks='auto') for file_ in file_names ]
    concatenated_arr = da.concatenate(dask_arrs, axis=0)
    if channel is not None:
        concatenated_arr = concatenated_arr[:,:,:,channel]
    return concatenated_arr

def find_int_log(int_path, att_path, percentile_level, channel=None):
    concatenated_att = concat_att(att_path, channel)
    concatenated_int = concat_data(int_path, channel)
    att_threshold = da.percentile(concatenated_att.flatten(), percentile_level)
    masked_int = concatenated_int[concatenated_att > att_threshold]
    masked_int = da.log(masked_int[masked_int > 0])
    masked_hist, masked_bins = da.histogram(masked_int, bins=20, range=(
       masked_int.min().compute(), masked_int.max().compute()), density=True)
    histogram = masked_hist.compute()
    return masked_int, att_threshold.compute(), masked_bins, histogram

def find_int(int_path, att_path, percentile_level, channel=None):
    concatenated_att = concat_att(att_path, channel)
    concatenated_int = concat_data(int_path, channel)
    att_threshold = da.percentile(concatenated_att.flatten(), percentile_level)
    masked_int = concatenated_int[concatenated_att > att_threshold]
    masked_hist, masked_bins = da.histogram(masked_int, bins=20, range=(
       masked_int.min().compute(), masked_int.max().compute()), density=True)
    histogram = masked_hist.compute()
    return att_threshold.compute(), masked_bins, histogram

def intensity_with_attribution():
    # import pdb; pdb.set_trace()
    from utils.aia_metadata import all_wavelengths
    npy_att_dir_path_1 = "/data/linn/tmp_op_v9_5c_allattribs/1/"
    npy_int_dir_path_1 = "/data/linn/tmp_1arbdfj4_allintensities/1/"


    npy_att_dir_path_0 = "/data/linn/tmp_op_v9_5c_allattribs/0/"
    npy_int_dir_path_0 = "/data/linn/tmp_1arbdfj4_allintensities/0/"
    # neglect 0 and negative values when taking the percentiles
    # since we neglect those when taking the log for plotting distribution?
    for percentile_level in (60, 80, 90, 99):

        # percentile_level = 60
        channel = 2
        # att_threshold_1, masked_bins_1, histogram_1 = find_int(npy_int_dir_path_1, npy_att_dir_path_1, percentile_level, channel)
        # att_threshold_0, masked_bins_0, histogram_0 = find_int(npy_int_dir_path_0, npy_att_dir_path_0, percentile_level, channel)

        masked_int_1, att_threshold_1, masked_bins_1, histogram_1 = find_int_log(npy_int_dir_path_1, npy_att_dir_path_1, percentile_level, channel)
        masked_int_0, att_threshold_0, masked_bins_0, histogram_0 = find_int_log(npy_int_dir_path_0, npy_att_dir_path_0, percentile_level, channel)
        median_1 = np.median(masked_int_1.compute())
        median_0 = np.median(masked_int_0.compute())
        ks_stat, p_val = ks_test(masked_int_0, masked_int_1)
        print("KS values", ks_stat, p_val)
        # masked_int = da.where(concatenated_att > p99_1, concatenated_int, np.nan)
        # concatenated_arr = da.log(concatenated_arr)
        #masked_int = np.random.choice(masked_int.flatten(), size=50000000)

        plt.figure(figsize=(10, 6))
        plt.bar(masked_bins_1[:-1], histogram_1, width=np.diff(masked_bins_1), edgecolor='black', label='flared')
        plt.bar(masked_bins_0[:-1], histogram_0, width=np.diff(masked_bins_0), edgecolor='black', label='non-flared', alpha=0.6)
        text = f"""percentile (flared):{att_threshold_1[0]:.4e} \n
        percentile (non-flared):{att_threshold_0[0]:.4e} \n
        ks-statistic:{ks_stat:.4e}, p-value:{p_val} \n
        median (flared):{median_1:.4f}, (non-flared):{median_0:.4f}"""

        anchored_text = AnchoredText(text, loc="upper left", prop=dict(size=8))
        anchored_text.patch.set_alpha(0.5)
        plt.gca().add_artist(anchored_text)
        plt.legend()
        plt.title(f"""Distribution of intensities for pixel attributions greater than {percentile_level} percentile for 
                  Passband:{all_wavelengths[channel]}""")
        plt.xlabel('Intensity')
        plt.ylabel('Normalized count')
        plt.grid(True)
        plt.savefig(f"intensity_histogram_with_attributions_{percentile_level}p_channel_{channel}.png", dpi=300)
        plt.close()

def process_class_npy(dir_path):
    file_names = glob.glob(dir_path+"*.npy")
    # print(file_names[0])
    dask_arrs = [ da.from_array(read_numpy(file_), chunks='auto') for file_ in file_names ]
    concatenated_arr = da.concatenate(dask_arrs, axis=0)
    # data_mean = concatenated_arr.mean().compute()
    # data_std = concatenated_arr.std().compute()
    # print(data_mean)
    # print(concatenated_arr.shape)
    concatenated_arr = concatenated_arr[:,2,:,:]
    concatenated_arr = concatenated_arr[concatenated_arr > 0]
    concatenated_arr = da.log(concatenated_arr)
    data_min = concatenated_arr.min().compute()
    data_99p = da.percentile(concatenated_arr.flatten(), 99).compute()
    print("99 percentile value of intensities", data_99p[0])

    intensity_hist, intensity_bins = da.histogram(concatenated_arr, bins=50, range=(data_min, data_99p[0]), density=True)
    # Trigger computation
    histogram = intensity_hist.compute()
    return concatenated_arr, intensity_bins, histogram
    # plt.plot(intensity_bins[:-1], histogram, color='blue', linestyle='-', linewidth=1)
    # plt.bar(intensity_bins[:-1], histogram, width=np.diff(intensity_bins), edgecolor='black')
    # plt.savefig("intensity_histogram_npy.png", dpi=300)

@delayed
def read_fits(file_path):
    with fits.open(file_path) as hdul:
        data = hdul[0].data.copy()
    del hdul[0].data
    return data

def process_class(dir_path):
    filenames = glob.glob(dir_path+"*")
    # print(filenames[0])
    patt=str(dir_path)+".*_171_.*"
    print("Looking for pattern", patt)
    matches = [ f for f in filenames if re.search(patt,f) is not None ]
    # print(matches[0])
    dask_arrays = [da.from_delayed(read_fits(path), shape=(512,512), dtype=np.dtype('>f8')) for path in matches]
    image_stack = da.stack(dask_arrays, axis=0)
    # Compute a global intensity histogram across resized images
    # intensity_hist, intensity_bins = da.histogram(image_stack, bins=256, range=(0, 255))
    data_mean = image_stack.mean().compute()
    data_std = image_stack.std().compute()
    print("Mean and std", data_mean, data_std)
    image_stack = (image_stack - data_mean)/data_std
    data_99p = da.percentile(image_stack.flatten(), 99).compute()
    print("99 percentile value of intensities", data_99p)
    # import sys; sys.exit(0)
    data_min = image_stack.min().compute()
    data_max = image_stack.max().compute()
    print("data min", data_min)
    print("data max", data_max)
    # intensity_hist, intensity_bins = da.histogram(image_stack, bins=256, range=(data_min, data_max))
    intensity_hist, intensity_bins = da.histogram(image_stack, bins=50, range=(data_min, data_99p[0]))
    # Trigger computation
    histogram = intensity_hist.compute()

    plt.figure(figsize=(10, 6))
    # plt.plot(intensity_bins[:-1], histogram, color='blue', linestyle='-', linewidth=1)
    plt.bar(intensity_bins[:-1], histogram, width=np.diff(intensity_bins), edgecolor='black')
    plt.title('Intensity Distribution Histogram')
    plt.xlabel('Intensity')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig("intensity_histogram.png", dpi=300)
    plt.show()

if __name__=="__main__":
    dir_path_1 = "/data/linn/E6_extracted_pos/"
    dir_path_2 = "/data/linn/E6_extracted_neg/"
    tracemalloc.start()
    #process_class(dir_path_1)
    npy_dir_path_1 = "/data/linn/tmp_1arbdfj4_allintensities/1/"
    npy_dir_path_0 = "/data/linn/tmp_1arbdfj4_allintensities/0/"
    # process_class_npy(npy_dir_path_1)
    #make_dist_plots()
    intensity_with_attribution()
    current, peak = tracemalloc.get_traced_memory()
    print(f"Peak: {peak/ 10**6}MB")
