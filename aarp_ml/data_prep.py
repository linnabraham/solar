import pandas as pd
from tqdm import tqdm
from datetime import datetime, timedelta
from glob import glob
from astropy.io import fits
import numpy as np
import concurrent.futures
import os
from PIL import Image

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

def read_7h_fits(fits_path):
    hdul = fits.open(fits_path)
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
            #print("Empty data encountered in hour number", hour_num, fits_path)
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
    return row

def summarize(shapes_7h:list, lons_7h:list, exp_7h:list, timestamps:list):
    stats = {}
    lons_7h_min = np.min(lons_7h)
    lons_7h_max = np.max(lons_7h)
    #assert any(np.isnan(x).any() for x in shapes_7h), "At least one value in one of the tuples in the list is np.nan"
    # Find the shape that maximizes the area of the patch
    shape_7h_max = shapes_7h[np.argmax(np.prod(shapes_7h, axis=1))]
    # Assume that timestamps are returned in the proper chronological order
    timestamp = timestamps[0]

    stats["min_lon"] = lons_7h_min
    stats["max_lon"] = lons_7h_max
    stats["max_height"] = shape_7h_max[0]
    stats["max_width"] = shape_7h_max[1]
    stats["start_time"] = timestamp

    return stats

def process_dir(dirpath, label):
    files = glob(dirpath)
    for file in tqdm(files):
        ret_val = read_7h_fits(file)
        if ret_val is not None:
            row = read_7h_fits(file)
            row["fits_fullpath"] = file
            row["label"] = label
            rows.append(row)

def gen_table_7h():
    pos_data = "/data/linn/newpipe_compressed/pos"
    neg_data = "/data/linn/newpipe_compressed/neg"

    rows = []
    for data_path, label in zip((pos_data, neg_data), (1,0)):
        pattern = f"{data_path}/*.fits"
        process_dir(pattern, label)

    table = pd.DataFrame(rows)
    return table

def select_7h(df):
    df.min_lon = df.min_lon.replace(-999999, np.nan)
    df.max_lon = df.max_lon.replace(-999999, np.nan)
    df = df[~df.min_lon.isna()]
    #print(df.max_lon.isna().sum())
    df = df[df.wavelength != 1600]
    df = df[(np.abs(df.min_lon) < 60) & (np.abs(df.max_lon) < 60)]
    return df

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

def resize_and_save_in_parallel(files_to_process, dest, targ_shape=(512,512)):

    # Number of parallel threads/workers
    num_threads = 15  # You can adjust this based on your system capabilities

    for file_path in tqdm(files_to_process):

        harpnum, wavelength, obs_start, timestamps, images = unpack_7h_fits(file_path)

        # Using ThreadPoolExecutor for parallel processing
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:

            # Map the resize_and_save function to each array in parallel
            pad_and_scaled = list(executor.map(downscale_and_pad, images, [targ_shape]*77))

            executor.map(save_to_fits, pad_and_scaled, [harpnum]*77, [wavelength]*77, [obs_start]*77, timestamps, [dest]*77)

def extract_7h(selected_7h, pos_data, neg_data):
    targ_shape = (512, 512)

    df = pd.read_csv(selected_7h)

    # make sure we are not extracting same file again
    assert len(pd.unique(df.fits_fullpath)) == len(df.fits_fullpath)

    print("Extracting 7h FITS observation into individual images")

    pos_files = df.fits_fullpath[df.label==1]
    print("Working on postive samples first")
    print("Files to extract", len(pos_files))
    dest = pos_data
    if not os.path.exists(dest):
        os.mkdir(dest)
    print("Saving to ", dest)
    resize_and_save_in_parallel(pos_files, dest=dest)

    neg_files = df.fits_fullpath[df.label==0]
    print("Working on negative samples")
    print("Files to extract", len(neg_files))
    dest = neg_data
    if not os.path.exists(dest):
        os.mkdir(dest)
    print("Saving to ", dest)
    resize_and_save_in_parallel(neg_files, dest=dest)

def pad_along_height(image, target_shape=(512,512)):
    qs_pixels = list(image[:2, :].flatten()) + list(image[-2:,:].flatten())+ \
        list(image[:,:2].flatten()) + list(image[:,-2:].flatten())
    aspect_ratio = image.shape[1]/image.shape[0]
    target_height, target_width = target_shape
    height_before_pad = int(target_width/aspect_ratio)
    #print("Height before pad", height_before_pad)
    height_diff = target_height - height_before_pad
    half_diff = height_diff // 2
    #print("Half diff", half_diff)
    black = np.zeros(target_shape)
    target_width = target_shape[1]
    resized_image = np.array(Image.fromarray(image).resize((target_width,
                                                            height_before_pad)))
    #print("shape of resized image", resized_image.shape)
    if height_diff % 2 == 0:
        black[half_diff:-half_diff,:] = resized_image
    else:
        black[half_diff:-(half_diff+1),:] = resized_image
        black[-half_diff-1,:] = resized_image [-1,:]


    black_top = black[:half_diff,:]
    top_qs = np.random.choice(qs_pixels, size=(black_top.shape))
    for row in np.arange(top_qs.shape[0]):
        row_from_bottom = top_qs.shape[0]-row
        top_qs_height = top_qs.shape[0]
        x = row_from_bottom/top_qs_height
        black_top[row_from_bottom-1, :] = top_qs[row_from_bottom-1,:] * (1-x) + \
                resized_image[0,:] * x

    black_bottom = black[-half_diff:,:] 
    bottom_qs = np.random.choice(qs_pixels, size=(black_bottom.shape))
    for row in np.arange(bottom_qs.shape[0]):
        row_from_bottom = bottom_qs.shape[0]-row
        bottom_qs_height = bottom_qs.shape[0]
        x = row_from_bottom/bottom_qs_height
        #black_bottom[-(row_from_bottom-1), :] = bottom_qs[-(row_from_bottom-1),:] * (1-x) + resized_image[-1:,:] * x
        black_bottom[-(row_from_bottom), :] = bottom_qs[-(row_from_bottom),:] * (1-x) + resized_image[-1:,:] * x

    return black

def pad_along_width(image, target_shape=(512,512)):
    qs_pixels = list(image[:2, :].flatten()) + list(image[-2:,:].flatten())+ \
        list(image[:,:2].flatten()) + list(image[:,-2:].flatten())
    aspect_ratio = image.shape[1]/image.shape[0]
    target_height, target_width = target_shape
    width_before_pad = int(target_height * aspect_ratio)
    #print("Width before pad", width_before_pad)
    width_diff = target_width - width_before_pad
    half_diff = width_diff // 2
    #print("Half diff", half_diff)
    black = np.zeros(target_shape)
    target_width = target_shape[1]
    resized_image = np.array(Image.fromarray(image).resize((width_before_pad,
                                                            target_height)))
    #print("shape of resized image", resized_image.shape)
    if width_diff % 2 == 0:
        black[:,half_diff:-half_diff] = resized_image
    else:
        black[:,half_diff:-(half_diff+1)] = resized_image
        black[:,-half_diff-1] = resized_image [:,-1]


    black_left = black[:,:half_diff]
    left_qs = np.random.choice(qs_pixels, size=(black_left.shape))
    for col in np.arange(left_qs.shape[1]):
        col_from_right = left_qs.shape[1]-col
        left_qs_width = left_qs.shape[1]
        x = col_from_right/left_qs_width
        black_left[:,col_from_right-1] = left_qs[:,col_from_right-1] * (1-x) + \
                resized_image[:,0] * x

    black_right = black[:,-half_diff:]
    right_qs = np.random.choice(qs_pixels, size=(black_right.shape))
    for col in np.arange(right_qs.shape[1]):
        col_from_right = right_qs.shape[1]-col
        right_qs_width = right_qs.shape[1]
        x = col_from_right/right_qs_width
        black_right[:,-col_from_right] = right_qs[:,-col_from_right] * (1-x) + \
                resized_image[:,-3] * x

    return black

def downscale_and_pad(image, target_shape=(512,512)):
    """
    Function to downsize image to specified size
    Resizing is attempted in an aspect ratio aware way
    The aspect ratio is computed and used to fix either the width or height.
    The difference in the other dimension is calculated and this dimension is filled using 
    the quiet sun background by sampling from all the edges of the image 2 pixels wide.
    If the difference is odd, one of the edges is retained as black.

    """
    target_height, target_width = target_shape
    aspect_ratio = image.shape[1]/image.shape[0]
    #assert aspect_ratio != 1.
    if aspect_ratio > 1:
        return pad_along_height(image, target_shape=target_shape)

    elif aspect_ratio < 1:
        return pad_along_width(image, target_shape=target_shape)

    else:
        resized = np.array(Image.fromarray(image).resize((target_width, target_height)))
        return resized


class data_prep:
    def __init__(self, goes_event_list, harp_to_noaa_map, aarps_full_urlist):
        self.goes_event_list = goes_event_list
        self.harp_to_noaa_map = harp_to_noaa_map
        self.aarps_full_urlist = aarps_full_urlist

    @property
    def goes_df(self):
        goes_df = pd.read_csv(self.goes_event_list, parse_dates=["event_date", "start_time", "peak_time", "end_time"])
        return goes_df

    @property
    def harps_with_noaa_df(self):
        harps_with_noaa_df = pd.read_csv(self.harp_to_noaa_map, delim_whitespace=True)
        return harps_with_noaa_df

    @property
    def aarps_full_df(self):
        aarps_full_df = pd.read_csv(self.aarps_full_urlist, header=None, names=['urls'])
        return aarps_full_df

    def get_clean_goes_df(self):
        harps_with_noaa_df = split_onmult(self.harps_with_noaa_df)
        goes_df = self.goes_df
        goes_df['harpnum'] = goes_df['noaa_active_region'].apply(match_noaa_to_harpnum, args=(harps_with_noaa_df,))

        # remove cases where there is no corresponding noaa AR number that matches
        goes_df  = goes_df.query("harpnum != -1")
        # select only AARPS that have resulted in major flares
        goes_df = goes_df[goes_df['goes_class'].apply(lambda x: x[0]) == "X"]
        return goes_df

    def get_selected_url_df(self):
        urldf = split_urllist(self.aarps_full_df, "urls")
        urldf['label'] = -99
        urldf['goes_matched_start'] = None
        goes_df = self.get_clean_goes_df()
        goes_df_org = goes_df.copy()

        for index, row in tqdm(urldf.iterrows(), total=len(urldf)):
            # import pdb; pdb.set_trace()
            obs_start = datetime.strptime(row['Datetime'], "%Y.%m.%d_%H:%M:%S")
            aarp_id = row['AARP']
            if not any(goes_df_org['harpnum'] == aarp_id):
                urldf.at[index, 'label'] = 0
            elif any((goes_df['harpnum'] == aarp_id) & (obs_start + timedelta(hours=7) < self.goes_df['start_time'])):
                matching_rows = goes_df[(goes_df['harpnum'] == aarp_id) & (obs_start + timedelta(hours=7) < goes_df['start_time'])]
                if not matching_rows.empty:
                    matched_start_time = matching_rows['start_time'].values[0]
                    urldf.at[index, 'goes_matched_start'] = matched_start_time
                    urldf.at[index, 'label'] = 1

        print("No. of 7hr AARP observation matches", (urldf['label']==1).sum())
        print("No. of unique AARPS", pd.unique(urldf[urldf['label']==1].AARP))

        # print(urldf[urldf['goes_matched_start'].notna()])
        urls_pos = urldf['urls'][urldf['label']==1]
        #TODO:incorporate grouping by wavelength and then shuffling to select the negative AARPS
        urls_neg = urldf['urls'][urldf['label']==0][-5000:]
        return urls_pos, urls_neg
