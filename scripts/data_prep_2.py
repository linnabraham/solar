from glob import glob
from astropy.io import fits
import numpy as np
from tqdm import tqdm
from PIL import Image
import os
import pandas as pd
import concurrent.futures
"""
2. Extract files as individual images with padding applied
"""
def pad_to_size(image, target_height, target_width):
    current_height, current_width = image.shape[:2]
    pad_height = target_height - current_height
    pad_width = target_width - current_width

    # Calculate padding for each side
    top = pad_height // 2
    bottom = pad_height - top
    left = pad_width // 2
    right = pad_width - left

    # Determine the number of channels (grayscale or color)
    if len(image.shape) == 2:  # Grayscale image
        return np.pad(image, ((top, bottom), (left, right)), mode='constant', constant_values=0)
    elif len(image.shape) == 3:  # Color image
        return np.pad(image, ((top, bottom), (left, right), (0, 0)), mode='constant', constant_values=0)
    else:
        raise ValueError("Unsupported image shape: {}".format(image.shape))

def pad_and_resize(image, biggest_shape, final_shape):
    target_height, target_width = biggest_shape
    final_height, final_width = final_shape
    image = pad_to_size(image, target_height, target_width)
    resized = np.array(Image.fromarray(image).resize((final_width, final_height)))
    return resized

def resize_and_save_in_parallel(files_to_process, dest, biggest_shape, targ_shape):

    # Number of parallel threads/workers
    num_threads = 15  # You can adjust this based on your system capabilities

    for file_path in tqdm(files_to_process):

        harpnum, wavelength, obs_start, timestamps, images = unpack_7h_fits(file_path)

        # Using ThreadPoolExecutor for parallel processing
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:

            # Map the resize_and_save function to each array in parallel
            try:
                pad_and_resized = list(executor.map(pad_and_resize, images, [biggest_shape]*77, [targ_shape]*77))
            except:
                print(f"Resize failed for {file_path}")
                continue

            executor.map(save_to_fits, pad_and_resized, [harpnum]*77, [wavelength]*77, [obs_start]*77, timestamps, [dest]*77)

def extract_7h(df, pos_data, neg_data, biggest_shape, target_shape):
    # df = pd.read_csv(selected_7h)
    # targ_shape = (512, 512)
    # make sure we are not extracting same file again
    assert len(pd.unique(df.fits_fullpath)) == len(df.fits_fullpath)

    print("Extracting 7h FITS observation into individual images")
    for dest, label in zip((pos_data, neg_data), (1,0)):

        files = df.fits_fullpath[df.label==label]
        print(f"Working on samples with label:{label} first")
        print("Files to extract", len(files))
        if not os.path.exists(dest):
            os.mkdir(dest)
        print("Saving to ", dest)
        resize_and_save_in_parallel(files, dest=dest, biggest_shape=biggest_shape, targ_shape=target_shape)

def read_7h_fits(fits_path):
    try:
        hdul = fits.open(fits_path)
    except Exception as e:
        print(f"Error reading FITS file: {e}")

    main_header = hdul[0].header

    harpnum = main_header['HARPNUM']
    wavelength = main_header['WAVELNTH']
    obs_start = main_header['T_START']

    shapes_7h = []
    lons_7h = []
    exp_7h = []
    timestamps = []

    for hour_num in range(1, main_header['NTIMES']+1):
        data = hdul[hour_num].data
        header = hdul[hour_num].header

        if data is None:
            return None

        else:

            if 'LON_FWT' in header:
                lon = header['LON_FWT']
                lons_7h.append(lon)
            if 'EXPTIME' in header:
                exp_time = header['EXPTIME']
                exp_7h.append(exp_time)

            nimgs = data.shape[0]
            burst_shapes = []
            burst_timestamps = []

            # iterate over 11 images in a single fits extension
            for nimg in range(nimgs):
                img = data[nimg]
                burst_shapes.append(img.shape)
                obstime_key = f"T_IMG{nimg:0>2d}"
                timestamp = header[obstime_key]
                timestamps.append(timestamp)
            burst_shape = burst_shapes[np.argmax(np.prod(burst_shapes, axis=1))]
            shapes_7h.append(burst_shape)
    row = summarize(shapes_7h, lons_7h, exp_7h, timestamps)
    row['harpnum'] = harpnum
    row['wavelength'] = wavelength
    hdul.close()
    return row

def process_class(fits_dir, label):
    files = glob(f"{fits_dir}/*.fits")
    rows = []
    for file in tqdm(files):
        try:
            row = read_7h_fits(file)
        except Exception as e:
            print(f"Could not read FITS file:{e}")
        else:
            if not row is None:
                row["fits_fullpath"] = file
                row["label"] = label
                rows.append(row)
    return rows

def gen_table_7h(pos_dir, neg_dir):
    rows = []
    for data_path, label in zip((pos_dir, neg_dir), (1,0)):
        rows_ = process_class(data_path, label)
        rows.extend(rows_)
    table = pd.DataFrame(rows)
    return table

def summarize(shapes_7h:list, lons_7h:list, exp_7h:list, timestamps:list):
    stats = {}
    lons_7h_min = np.min(lons_7h)
    lons_7h_max = np.max(lons_7h)
    shape_7h_max = shapes_7h[np.argmax(np.prod(shapes_7h, axis=1))]
    timestamp = timestamps[0]

    stats["min_lon"] = lons_7h_min
    stats["max_lon"] = lons_7h_max
    stats["max_height"] = shape_7h_max[0]
    stats["max_width"] = shape_7h_max[1]
    #stats["start_time"] = timestamp

    return stats

def unpack_7h_fits(fits_path):
    hdul = fits.open(fits_path)
    main_header = hdul[0].header
    wavelength = main_header['WAVELNTH']
    harpnum = main_header['HARPNUM']
    obs_start = main_header['T_START']
    images = []
    timestamps = []
    for hour_num in range(1,main_header['NTIMES']+1):
        data = hdul[hour_num].data
        if data is None:
            print("Empty data encountered in hour number",hour_num, fits_path)
            continue
            #return None
        header = hdul[hour_num].header
        extname = f"T_IMG{hour_num:0>2d}"
        nimgs = data.shape[0]
        # iterate over 11 images in a single fits extension
        for nimg in range(nimgs):
            img = data[nimg]
            obstime_key = f"T_IMG{nimg:0>2d}"
            timestamp = header[obstime_key]
            if timestamp == 'NaN':
                print("timstamp missing in header", obstime_key, fits_path)
                #return None
                continue
            images.append(img)
            timestamps.append(timestamp)
    return harpnum, wavelength, obs_start, timestamps, images

def save_to_fits(image, harpnum, wavelength, obs_start, timestamp, dest):
    hdu = fits.PrimaryHDU(data=image)
    new_hdul = fits.HDUList([hdu])
    fits_filename = f'{harpnum}_{wavelength}_{obs_start}_{timestamp}.fits'
    new_hdul.writeto(os.path.join(dest,fits_filename))

def split_urllist(df, name):
    """
    Using regex matching convert the url paths into seperate columns
    """
    urldf = pd.DataFrame({'urls': df[name]})
    urldf[['Datetime', 'AARP', 'Wavelength']] = \
    df[name].str.extract(r'(\d{4}\.\d{2}\.\d{2}_\d{2}:\d{2}:\d{2})_7h@1h_AARP(\d+)_(\d+)\.fits')
    urldf['AARP'] = urldf['AARP'].astype(int)
    urldf['Wavelength'] = urldf['Wavelength'].astype(int)

    return urldf

def remove_offlimb(df, lon_threshold=60):
    df.min_lon = df.min_lon.replace(-999999, np.nan)
    df.max_lon = df.max_lon.replace(-999999, np.nan)
    print(f"length of df {len(df)}")
    grouped = df.groupby(["Datetime","harpnum"])
    print(f"Found {grouped.ngroups} groups")
    concat_list = []
    for group_key, group_df in tqdm(grouped):
            #assert len(group_df) ==7
            if not len(group_df) == 7:
                print(f"Found group with length not equal to 7 for haprnum: {group_key[1]}")
            group_df["max_abs_lon"]  = group_df[["min_lon", "max_lon"]].abs().max(axis=1)
            if len(group_df[group_df.max_abs_lon > lon_threshold]) > 0:
                continue
            else:
                concat_list.append(group_df)
    return pd.concat(concat_list)

if __name__=="__main__":

    pos_dir_7h = "/data/linn/E8/compressed/pos"
    neg_dir_7h = "/data/linn/E8/compressed/neg"
    table_7h = gen_table_7h(pos_dir_7h, neg_dir_7h)
    table_7h[["Datetime", "AARP", "Wavelength"]] = split_urllist(table_7h, "fits_fullpath")[["Datetime", "AARP", "Wavelength"]]
    table_7h_clean = remove_offlimb(table_7h)
    biggest_height = table_7h_clean.max_height.max()
    biggest_width = table_7h_clean.max_width.max()
    biggest_shape = (biggest_height, biggest_width)

    pos_dir_single = "/data/linn/E8/extracted/pos"
    neg_dir_single = "/data/linn/E8/extracted/neg"
    extract_7h(table_7h_clean, pos_dir_single, neg_dir_single, biggest_shape, target_shape=(512,512))
