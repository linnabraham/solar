#!/bin/env python

"""
This script reads the table previously saved and processes it
to remove things like AARPS with observations on the limb
"""

import pandas as pd
from tqdm import tqdm
import numpy as np
from PIL import Image
import concurrent.futures
from astropy.io import fits
import os
import argparse

def make_padded(targ_shape, img):
    y = (targ_shape[0] - img.shape[0])//2
    yy = targ_shape[0] - y -img.shape[0]
    x = (targ_shape[1] - img.shape[1])//2
    xx = targ_shape[1] - x -img.shape[1]
    #print("Input img shape", img.shape)
    #print(y, yy, x, xx)
    if (y<0) or (yy<0) or (x<0) or (xx<0):
        #print("Received negative value")
        return img
    padded = np.pad(img,((y,yy),(x,xx)), mode='edge')
    return padded

def pad_scale(img, pad_shape, targ_shape):
    padded = make_padded(pad_shape, img)
    resized = resize(padded, targ_shape)
    return resized

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
    #org_file_path = os.path.basename(fits_path)
        #table = create_table(harpnum, wavelength, obs_start, fits_path, timestamps, images)
    return harpnum, wavelength, obs_start, timestamps, images

#def resize(image:np.ndarray):
#    global height
#    global width
#    image = np.array(Image.fromarray(image).resize((height, width), Image.NEAREST))
#    return image

def resize(img, targ_shape):
    output = np.array(Image.fromarray(img).resize(targ_shape))
    return output

def save_to_fits(image, harpnum, wavelength, timestamp, dest):
    global height
    global width
    assert image.shape[0] == height
    assert image.shape[1] == width
    hdu = fits.PrimaryHDU(data=image)
    new_hdul = fits.HDUList([hdu])
    fits_filename = f'{harpnum}_{wavelength}_{timestamp}.fits'
    #print("Saving file to ", os.path.join(dest,fits_filename))
    new_hdul.writeto(os.path.join(dest,fits_filename))

def remove_off_disk(df, lon_threshold=60):
    print("Removing off disk ")
    grouped = df.groupby(['harpnum','timestamp'])
    concat_list = []
    for group_key, group_df in tqdm(grouped):
        if len(group_df) ==7:
            group_harpnum = int(group_key[0])
            group_timestamp = group_key[1]
            lon = group_df['longitude'].iloc[0]
            if not np.abs(lon) < lon_threshold:
                continue
            else:
                concat_list.append(group_df)

    return concat_list

def simultaneous_multiband(df):
    """
    Accepts a dataframe with each row referring to a single AARP observation
    Return a generator that generates groups of observations
    with the same timestamp and harp id but different wavelengths
    """
    grouped = df.groupby(['harpnum','timestamp'])
    for group_key, group_df in grouped:
        if len(group_df) ==7:
            group_harpnum = int(group_key[0])
            group_timestamp = group_key[1]
            yield group_df

def process(csv_path):
    master_df = pd.read_csv(csv_path, index_col=[0])

    # dont use 1600 for now because the timestamps are different wrt other wavelengths
    master_df = master_df[master_df.wavelength!=1600]

    concat_list  = remove_off_disk(master_df)
    #print("No. of observations", len(concat_list))

    concat_df = pd.concat(concat_list)
    #print("No. of unique harps present", len(pd.unique(concat_df.harpnum)))

    df_med = concat_df[(concat_df.img_height.between(400,800)) & (concat_df.img_width.between(400,800))]
    
    return df_med

def filter_size(df, low_dims, high_dims):

    desired_size = df[(df['dimensions'].apply(lambda x: eval(x)[0]).between(low_dims[0],high_dims[0])) & (df['dimensions'].apply(lambda x: eval(x)[1]).between(low_dims[1],high_dims[1]))]

    return desired_size


def resize_and_save_in_parallel(files_to_process, dest, pad_shape, targ_shape=(512,512)):

    # Number of parallel threads/workers
    num_threads = 15  # You can adjust this based on your system capabilities

    for file_path in tqdm(files_to_process):

        harpnum, wavelength, timestamps, images = unpack_7h_fits(file_path)

        # Using ThreadPoolExecutor for parallel processing
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:

            # Map the resize_and_save function to each array in parallel
            #pad_or_scaled = list(executor.map(pad_or_scale, images))
            pad_and_scaled = list(executor.map(pad_scale, images, [pad_shape]*77, [targ_shape]*77))
            #resized_images = list(executor.map(resize, pad_or_scaled))
            #print(len(pad_and_scaled))

            #print(dest)
            #print("Writing files to disk")
            executor.map(save_to_fits, pad_and_scaled, [harpnum]*77, [wavelength]*77, timestamps, [dest]*77)

            #break
            #import sys
            #sys.exit(0)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-input_shape', nargs='+', type=int, default=(512,512))
    args = parser.parse_args()

    height = args.input_shape[0]
    width = args.input_shape[1]

    pos_table_path = "data/extracted_aarps_pos_details.csv"
    neg_table_path = "data/extracted_aarps_neg_details.csv"

    concat_df = process(pos_table_path)

    files_to_process = pd.unique(concat_df['org_file_fullpath'])
    #print("Files to process", files_to_process)

    resize_and_save_in_parallel(files_to_process, dest = "/data/linn/newpipe_extracted_pos_new")

    print("No. of observations", len(concat_df))
    print("No. of unique harps present", len(pd.unique(concat_df.harpnum)))
    print("No. of unique 7h FITS files after filtering with fixed dimensions", len(pd.unique(concat_df['org_file_fullpath'])))

    base_path = "/data/linn/newpipe_extracted_pos_new/"
    concat_df["single_img_path"] = concat_df.apply(lambda x: os.path.join(base_path,f"{x[1]}_{x[2]}_{x[5]}.fits"), axis=1)
    concat_df.to_csv("data/pos_traindata_E3_new.csv", index=False)



    # do for negative
    #desired_size = process(neg_table_path, low_dim=200, high_dim=700)
    #desired_size = process(neg_table_path, low_dim=200, high_dim=600)
    concat_df = process(neg_table_path)

    low_dims=(323,190)
    high_dims=(816, 1444)

    #percentage_reduction = 0.1
    #low_dims = tuple(round(element * (1 - percentage_reduction)) for element in low_dims)

    #percentage_increase = 0.1
    #high_dims = tuple(round(element * (1 + percentage_increase)) for element in high_dims)

    print(low_dims)
    print(high_dims)

    #desired_size = filter_size(concat_df, low_dims=(323,190), high_dims=(816, 1444))
    #desired_size = concat_df

    print("No. of observations", len(concat_df))
    print("No. of unique harps present", len(pd.unique(concat_df.harpnum)))
    print("No. of unique 7h FITS files after filtering with fixed dimensions", len(pd.unique(concat_df['org_file_fullpath'])))

    #print(desired_size)
    #print(desired_size['dimensions'])
    #files_to_process = pd.unique(desired_size['file_full_path'])
    files_to_process = pd.unique(concat_df['org_file_fullpath'])
    # no error is raised if this dir doesnt exist

    resize_and_save_in_parallel(files_to_process, dest = "/data/linn/newpipe_extracted_neg_new")


    base_path = "/data/linn/newpipe_extracted_neg_new/"
    #desired_size["single_img_path"] = desired_size.apply(lambda x: os.path.join(base_path,f"{x[1]}_{x[2]}_{x[5]}.fits"), axis=1)
    concat_df["single_img_path"] = concat_df.apply(lambda x: os.path.join(base_path,f"{x[1]}_{x[2]}_{x[5]}.fits"), axis=1)
    concat_df.to_csv("data/neg_traindata_E3_new.csv", index=False)
