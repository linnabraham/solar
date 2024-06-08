#!/bin/env python
"""
Script that reads the 7h fits files into individual images
and does the desired padding to a fixed shape and saves to disk
as individual fits files with a suitably encoded filename
Multithreading is used for parallelization
"""
import argparse
import pandas as pd
import os
import numpy as np
from tqdm import tqdm
import concurrent.futures
from astropy.io import fits
from simple_padding import downscale_and_pad

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

if __name__=="__main__":
    parser = argparse.ArgumentParser() 
    parser.add_argument("-pos-data", "--pos-data")
    parser.add_argument("-neg-data", "--neg-data")
    args = parser.parse_args()

    selected_7h = "data/selected_7h.csv"
    targ_shape = (512, 512)

    df = pd.read_csv(selected_7h)

    # make sure we are not extracting same file again
    assert len(pd.unique(df.fits_fullpath)) == len(df.fits_fullpath)

    print("Extracting 7h FITS observation into individual images")

    pos_files = df.fits_fullpath[df.label==1]
    print("Working on postive samples first")
    print("Files to extract", len(pos_files))
    dest = args.pos_data
    if not os.path.exists(dest):
        os.mkdir(dest)
    print("Saving to ", dest)
    resize_and_save_in_parallel(pos_files, dest=dest)

    neg_files = df.fits_fullpath[df.label==0]
    print("Working on negative samples")
    print("Files to extract", len(neg_files))
    dest = args.neg_data
    if not os.path.exists(dest):
        os.mkdir(dest)
    print("Saving to ", dest)
    resize_and_save_in_parallel(neg_files, dest=dest)
