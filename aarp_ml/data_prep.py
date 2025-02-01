from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from glob import glob
from astropy.io import fits
from tqdm import tqdm
from PIL import Image
import os
import concurrent.futures
import json
import re
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KernelDensity
from joblib import Memory
np.random.seed(42)
memory = Memory(location='/data/linn/cachedir', verbose=0)

"""
Scripts used for data download and processing
without using any class functions
"""

def select_urls(urldf, goes_df):
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

def split_onmult(df):
    """
    If there are multiple harpnums matching a single NOAA_ARS number turn those into extra rows
    """
    df = df.assign(NOAA_ARS=df['NOAA_ARS'].str.split(',')).explode('NOAA_ARS').reset_index(drop=True)
    df['NOAA_ARS'] = df['NOAA_ARS'].astype(int)

    return df

def match_noaa_to_harpnum(x, harps_with_noaa_df):
    """
    Match each noaa number to a harpnum
    If not found return -1
    """
    match = harps_with_noaa_df['HARPNUM'].loc[harps_with_noaa_df['NOAA_ARS']==x].values
    if len(match)==0:
        return -1
    else:
        return match[0]

def label_urls(urldf, goes_df):
        urldf['label'] = -99
        flared_aarp_ids = set(goes_df.harpnum)
        for index, row in tqdm(urldf.iterrows(), total=len(urldf)):
            aarp_id = row['AARP']
            if not any(goes_df['harpnum'] == aarp_id):
                urldf.at[index, 'label'] = 0
            elif any(goes_df['harpnum'] == aarp_id):
                urldf.at[index, 'label'] = 1
        return urldf

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

def select_neg_urls(urldf, num_aarps):
    urldf = urldf.sample(frac=1).reset_index(drop=True)
    neg_aarp_ids = urldf.AARP.unique()[:num_aarps]
    urldf = urldf[urldf.AARP.isin(neg_aarp_ids)]

    group_keys = list(urldf.groupby(["Datetime", "AARP"]).groups.keys())
    np.random.shuffle(group_keys)
    collected_groups = []

    for key in group_keys:
        df = urldf.groupby(["Datetime", "AARP"]).get_group(key)
        collected_groups.append(df)

    urldf = pd.concat(collected_groups, ignore_index=True)
    return urldf

@memory.cache
def get_download_list(goes_event_list, aarp_full_urls, harp_to_noaa):
    """
    Create list of files to download after applying certain
    selections
    """
    goes_df = pd.read_csv(goes_event_list, parse_dates=["event_date", "start_time", "peak_time", "end_time"])

    aarps_full_df = pd.read_csv(aarp_full_urls, header=None, names=['urls'])
    aarps_clean_df = split_urllist(aarps_full_df, "urls")
    aarps_clean_df = aarps_clean_df[aarps_clean_df["Wavelength"] != 1600]

    harp_to_noaa_df =pd.read_csv(harp_to_noaa, delim_whitespace=True)
    harp_to_noaa_df = split_onmult(harp_to_noaa_df)

    goes_df['harpnum'] = goes_df.noaa_active_region.apply(match_noaa_to_harpnum, args=(harp_to_noaa_df,))
    goes_df = goes_df[goes_df.harpnum != -1]

    urldf = label_urls(aarps_clean_df, goes_df)
    pos_urls = urldf[urldf.label==1]

    selected_urls = select_urls(pos_urls, goes_df)

    matched_urls = selected_urls[selected_urls.goes_matched_start.notna()]
    print(f"URLs with matches:", len(matched_urls))

    pos_urls = selected_urls[~selected_urls.goes_matched_start.notna()]
    print(f"URLs in positive class:", len(pos_urls))

    neg_urls = urldf[urldf.label==0]
    num_aarps = pos_urls.AARP.nunique()*4
    neg_urls = select_neg_urls(neg_urls, num_aarps)
    print(f"URLs in negative class:", len(matched_urls))
    return pos_urls, neg_urls

def extract_7h(df, padding_func, pos_data, neg_data, biggest_shape, target_shape):
    # make sure we are not extracting same file again
    assert len(pd.unique(df.fits_fullpath)) == len(df.fits_fullpath)

    print("Extracting 7h FITS observation into individual images")
    for dest, label in zip((pos_data, neg_data), (1,0)):

        files = df.fits_fullpath[df.Label==label]
        print(f"Working on samples with label:{label} first")
        print("Files to extract", len(files))
        if not os.path.exists(dest):
            os.mkdir(dest)
        print("Saving to ", dest)
        pad_and_resize_in_parallel(files, padding_func, dest=dest, biggest_shape=biggest_shape, targ_shape=target_shape)

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
                continue
            images.append(img)
            timestamps.append(timestamp)
    return harpnum, wavelength, obs_start, timestamps, images

def save_to_fits(image, harpnum, wavelength, obs_start, timestamp, dest):
    hdu = fits.PrimaryHDU(data=image)
    new_hdul = fits.HDUList([hdu])
    fits_filename = f'{harpnum}_{wavelength}_{obs_start}_{timestamp}.fits'
    new_hdul.writeto(os.path.join(dest,fits_filename))

def remove_offlimb(df, lon_threshold=60):
    print(f"length of df {len(df)}")
    grouped = df.groupby(["Datetime","AARP","Wavelength"])
    print(f"Found {grouped.ngroups} groups")
    concat_list = []
    for group_key, group_df in tqdm(grouped):
            group_len = len(group_df)
            if not len(group_df) == 77:
                print(f"Found group with length {group_len} not equal to 77 for haprnum: {group_key[1]}, Skipping....")
                continue
            group_df["max_abs_lon"] = group_df[["Longitude"]].abs().max(axis=1)
            if len(group_df[group_df.max_abs_lon > lon_threshold]) > 0:
                continue
            else:
                concat_list.append(group_df)
    return pd.concat(concat_list)

def split_data(harpnums: pd.Series):
    """
    Given a list of harp ids find the unique ids and split it into train, val and test
    so that no two sets have the same harp id
    """
    unique_harps = pd.unique(harpnums)
    train_data, test_data = train_test_split(unique_harps, test_size=0.2, random_state=42)
    train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)
    aarp_lists = (train_data, val_data, test_data)
    return aarp_lists

def simultaneous_multiband(df):
    """
    Accepts a dataframe with each row referring to a single AARP observation
    Return a generator that generates groups of observations
    with the same timestamp and harp id but different wavelengths
    """
    fixed_bands = [94, 131, 171, 193, 211, 304, 335]
    grouped = df.groupby(['harpnum','timestamp'])
    for group_key, group_df in grouped:
        if len(group_df) == 7:
            assert set([int(item) for item in group_df['wavelength'].values]) == set(fixed_bands)
            group_harpnum = int(group_key[0])
            group_timestamp = group_key[1]
            yield (group_harpnum, group_timestamp, group_df)

def dir_to_json(extracted_dest_pos, extracted_dest_neg, filename):
    """
    Create a training metadata file as json
    """
    training_full = []
    validation_full = []
    test_full = []

    for label, extracted_path in zip((1,0), (extracted_dest_pos, extracted_dest_neg)):
        training, validation, test = dir_to_dataset(extracted_path, label=label)
        training_full.extend(training)
        validation_full.extend(validation)
        test_full.extend(test)

    fixed_bands = [94, 131, 171, 193, 211, 304, 335]

    metadata = { "name" : "Fixed size AARPS",
            "description" : "Active Region patches from AARPS database downscaled or padded to a fixed resolution and unpacked",
            "channels" : {
                "0" : fixed_bands[0],
                "1" : fixed_bands[1],
                "2" : fixed_bands[2],
                "3" : fixed_bands[3],
                "4" : fixed_bands[4],
                "5" : fixed_bands[5],
                "6" : fixed_bands[6]
                },
            "training" : training_full,
            "validation" : validation_full,
            "test" : test
            }
    pretty = json.dumps(metadata, indent=4)

    with open(filename, "w") as write_file:
        json.dump(metadata, write_file, indent=4)

def summarize_shapes(table_7h_clean):
    data = {}
    for label in (0,1):
        max_height = table_7h_clean[table_7h_clean.Label==label].img_height.max()
        max_width = table_7h_clean[table_7h_clean.Label==label].img_width.max()
        min_height = table_7h_clean[table_7h_clean.Label==label].img_height.min()
        min_width = table_7h_clean[table_7h_clean.Label==label].img_width.min()
        mean_width = table_7h_clean[table_7h_clean.Label==label].img_width.mean()
        std_width = table_7h_clean[table_7h_clean.Label==label].img_width.std()
        mean_height = table_7h_clean[table_7h_clean.Label==label].img_height.mean()
        std_height = table_7h_clean[table_7h_clean.Label==label].img_height.std()

        data_class = {"max_height":max_height, "max_width": max_width, "min_height":min_height, "min_width":min_width, 
                     "mean_width":mean_width, "std_width": std_width, "mean_height":mean_height, "std_height":std_height}
        data[f"{label}"]=data_class
    return data

def get_fov_limits(table_7h_clean):
    data = summarize_shapes(table_7h_clean)
    pos_mean_height = data.get("1").get("mean_height")
    pos_mean_width = data.get("1").get("mean_width")
    pos_std_height = data.get("1").get("std_height")
    pos_std_width = data.get("1").get("std_width")

    low_dims = pos_mean_height - pos_std_height, pos_mean_width - pos_std_width
    high_dims = pos_mean_height + pos_std_height, pos_mean_width + pos_std_width
    pos_max_height = data.get("1").get("max_height")
    pos_max_width = data.get("1").get("max_width")

    pos_min_height = data.get("1").get("min_height")
    pos_min_width = data.get("1").get("min_width")

    low_dims = pos_min_height, pos_min_width
    high_dims = pos_max_height, pos_max_width
    return low_dims, high_dims

def apply_shape_limits(table_7h_clean, low_dims, high_dims):
    neg_df = table_7h_clean[table_7h_clean.Label==0]
    pos_df = table_7h_clean[table_7h_clean.Label==1]
    neg_df = neg_df[(neg_df.img_height.between(low_dims[0], high_dims[0]) & neg_df.img_width.between(low_dims[1], high_dims[1]))]
    table_7h_clean = pd.concat([neg_df, pos_df])
    return table_7h_clean

def bias_analysis(table):
    na_val = table.min_lon.min()
    df = table[table.min_lon != na_val]
    numerical_features = ['max_height', 'max_width', 'min_lon', 'max_lon']
    for feature in numerical_features:
        if feature in df.columns:
            plt.figure(figsize=(10, 6))
            sns.histplot(data=df, x=feature, hue='label', kde=True, palette='Set1', bins=30)
            plt.title(f'Distribution of {feature} by Label')
            plt.show()

def pad_and_scale(image, padding_func, target_shape, final_shape=(512,512)):
    image = padding_func(image, target_shape=target_shape)
    target_height, target_width = final_shape
    return np.array(Image.fromarray(image).resize((target_width, target_height)))

def create_pad_and_scale(padding_function):
    """Returns a version of pad_and_scale with a predefined padding function."""
    def wrapped_pad_and_scale(image, target_shape, final_shape=(512, 512)):
        return pad_and_scale(image, padding_function, target_shape, final_shape)
    return wrapped_pad_and_scale

def pad_with_quiet(image, target_shape):
    """
    Pads an image with values randomly sampled from whole image to attain the target shape.

    Parameters:
        image (numpy.ndarray): Input image as a NumPy array.
        target_shape (tuple): Target shape as (target_height, target_width).

    Returns:
        numpy.ndarray: Padded image with the target shape.
    """
    # Ensure the target shape is valid
    target_height, target_width = target_shape
    img_height, img_width = image.shape[:2]

    if target_height < img_height or target_width < img_width:
        raise ValueError("Target shape must be greater than or equal to the image shape.")

    # Compute padding sizes
    pad_top = (target_height - img_height) // 2
    pad_bottom = target_height - img_height - pad_top
    pad_left = (target_width - img_width) // 2
    pad_right = target_width - img_width - pad_left

    # Sample values from the borders
    top_pad = np.random.choice(image.flatten(), (pad_top, img_width, image.shape[2] if image.ndim == 3 else 1))
    bottom_pad = np.random.choice(image.flatten(), (pad_bottom, img_width, image.shape[2] if image.ndim == 3 else 1))
    left_pad = np.random.choice(image.flatten(), (target_height, pad_left, image.shape[2] if image.ndim == 3 else 1))
    right_pad = np.random.choice(image.flatten(), (target_height, pad_right, image.shape[2] if image.ndim == 3 else 1))

    # Adjust dimensions if the image is grayscale
    if image.ndim == 2:
        top_pad = top_pad.squeeze()
        bottom_pad = bottom_pad.squeeze()
        left_pad = left_pad.squeeze()
        right_pad = right_pad.squeeze()

    # Pad the image
    padded_image = np.zeros((target_height, target_width, image.shape[2] if image.ndim == 3 else 1), dtype=image.dtype)
    if image.ndim == 2:
        padded_image = padded_image.squeeze()

    # Insert the original image into the center
    padded_image[pad_top:pad_top + img_height, pad_left:pad_left + img_width] = image

    # Fill top and bottom padding
    padded_image[:pad_top, pad_left:pad_left + img_width] = top_pad
    padded_image[pad_top + img_height:, pad_left:pad_left + img_width] = bottom_pad

    # Fill left and right padding
    padded_image[:, :pad_left] = left_pad
    padded_image[:, pad_left + img_width:] = right_pad

    return padded_image

def pad_and_resize_in_parallel(files_to_process, padding_func, dest, biggest_shape, targ_shape):
    # Number of parallel threads/workers
    num_threads = 15  # You can adjust this based on your system capabilities
    custom_pad_and_scale = create_pad_and_scale(padding_func)
    for file_path in tqdm(files_to_process):

        harpnum, wavelength, obs_start, timestamps, images = unpack_7h_fits(file_path)

        # Using ThreadPoolExecutor for parallel processing
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:

            # Map the resize_and_save function to each array in parallel
            try:
                pad_and_resized = list(executor.map(custom_pad_and_scale, images, [biggest_shape]*77, [targ_shape]*77))
            except Exception as e:
                print(f"Resize failed for {file_path}")
                print(f"{e} \n but image has shape {images[0].shape} and target is {biggest_shape}")
                continue

            executor.map(save_to_fits, pad_and_resized, [harpnum]*77, [wavelength]*77, [obs_start]*77, timestamps, [dest]*77)

def resample_on_shapes(shape_limited_df):
    positive_class = shape_limited_df[shape_limited_df["Label"] == 1]
    negative_class = shape_limited_df[shape_limited_df["Label"] == 0]
    grouped = positive_class.groupby(["Datetime", "AARP"])
    first_row_values_pos = grouped.first().reset_index()
    grouped = negative_class.groupby(["Datetime", "AARP"])
    first_row_values_neg = grouped.first().reset_index()
    positive_features = first_row_values_pos[['max_height', 'max_width']]
    kde = KernelDensity(kernel='gaussian', bandwidth=10)  # Adjust bandwidth as needed
    kde.fit(positive_features)

    negative_features = first_row_values_neg[['max_height', 'max_width']]
    first_row_values_neg['density_score'] = np.exp(kde.score_samples(negative_features))
    sampled_negatives = first_row_values_neg.sample(
        n=len(first_row_values_neg),  # Match the number of positive samples
        weights='density_score',  # Use density as sampling weight
        random_state=42          # For reproducibility
    )
    concat_sampled = pd.concat([first_row_values_pos, sampled_negatives.drop("density_score", axis=1)])
    selected_7h_df = pd.merge(
        shape_limited_df,  # Original dataset
        concat_sampled[["Datetime", "AARP"]],  # Filtered group identifiers
        on=["Datetime", "AARP"],  # Columns to match
        how="inner"  # Keep only matching rows
    )
    return selected_7h_df

def process_image_hdu(hdu, hdu_index, fits_fullpath, AARP, wavelength, datetime, label):
    """
    Function to extract metadata and image properties from an HDU
    """
    header = hdu.header
    hdu_rows = []
    obs_start = header.get('T_START', None)
    lon = header.get("LON_FWT", None)
    exp_time = header.get("EXPTIME", None)

    # Extract image data properties (if present)
    if hdu.data is not None:
        num_images = hdu.data.shape[0]
        for img_index in range(num_images):
            image_data = hdu.data[img_index]
            image_time = header.get(f"T_IMG{img_index:02d}", None)
            hdu_rows.append({
        "fits_fullpath": fits_fullpath,
        "AARP": AARP,
        "Wavelength": wavelength,
        "Datetime": datetime,
        "Label": label,
        "HDU_index": hdu_index,
        "img_time": image_time,
        "img_height": image_data.shape[0],
        "img_width": image_data.shape[1],
        "Longitude": lon,
        "Exposure": exp_time,
    }    )

    return hdu_rows

def read_from_disk(pos_dir, neg_dir):
    """
    Create a table using the FITS files from the positive and negative class directories
    """
    rows = []
    for data_path, label in zip((pos_dir, neg_dir), (1,0)):
        files = glob(f"{data_path}/*.fits")
        for file in tqdm(files):
            rows.append({"fits_fullpath":file, "label":label})
    table_on_disk = pd.DataFrame(rows)
    return table_on_disk

def process_table_on_disk(table_on_disk):
    """
    Input: DataFrame where each row corresponds to a downloaded FITS file
    and columns are made from the filename
    Ouput: DataFrame where each row corresponds to a single image in any of the FITS files
    """
    combined_data = []
    for _, row in table_on_disk.iterrows():
        fits_fullpath = row['fits_fullpath']
        AARP = row['AARP']
        wavelength = row['Wavelength']
        label = row['label']
        datetime = row['Datetime']
        hdu_rows = []
        try:
            with fits.open(fits_fullpath) as hdulist:
                assert hdulist is not None
                for hdu_index, hdu in enumerate(hdulist):
                    if isinstance(hdu, fits.PrimaryHDU):
                        continue
                    elif isinstance(hdu, fits.ImageHDU):
                        try:
                            image_hdu_rows = process_image_hdu(hdu, hdu_index, fits_fullpath, AARP, wavelength, datetime, label)
                        except Exception as e:
                            print(f"Error processing image hdu: {e}")
                        else:
                            hdu_rows.extend(image_hdu_rows)
        except Exception as e:
            print(f"Error processing file {fits_fullpath}: {e}")
        else:
            combined_data.extend(hdu_rows)
        combined_df = pd.DataFrame(combined_data)
    return combined_df
