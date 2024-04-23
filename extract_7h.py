#!/bin/env python
"""
Script that reads the 7h fits files into individual images
and does the desired padding to a fixed shape and saves to disk
as individual fits files with a suitably encoded filename
Multithreading is used for parallelization
"""
import pandas as pd
import os
import numpy as np
from tqdm import tqdm
import concurrent.futures
from astropy.io import fits
#from modified_pipeline_part3new import resize_and_save_in_parallel
from modified_pipeline_part3new import unpack_7h_fits, pad_scale
from simple_padding import downscale_and_pad

def save_to_fits(image, harpnum, wavelength, obs_start, timestamp, dest):
    hdu = fits.PrimaryHDU(data=image)
    new_hdul = fits.HDUList([hdu])
    fits_filename = f'{harpnum}_{wavelength}_{obs_start}_{timestamp}.fits'
    #print("Saving file to ", os.path.join(dest,fits_filename))
    new_hdul.writeto(os.path.join(dest,fits_filename))

def resize_and_save_in_parallel(files_to_process, dest, targ_shape=(512,512)):

    # Number of parallel threads/workers
    num_threads = 15  # You can adjust this based on your system capabilities

    for file_path in tqdm(files_to_process):

        harpnum, wavelength, obs_start, timestamps, images = unpack_7h_fits(file_path)

        # Using ThreadPoolExecutor for parallel processing
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:

            # Map the resize_and_save function to each array in parallel
            #pad_or_scaled = list(executor.map(pad_or_scale, images))
            #pad_and_scaled = list(executor.map(pad_scale, images, [pad_shape]*77, [targ_shape]*77))
            pad_and_scaled = list(executor.map(downscale_and_pad, images, [targ_shape]*77))
            #resized_images = list(executor.map(resize, pad_or_scaled))
            #print(len(pad_and_scaled))

            #print(dest)
            #print("Writing files to disk")
            executor.map(save_to_fits, pad_and_scaled, [harpnum]*77, [wavelength]*77, [obs_start]*77, timestamps, [dest]*77)

if __name__=="__main__":
    selected_7h = "data/selected_7h.csv"
    targ_shape = (512, 512)

    df = pd.read_csv(selected_7h)

    ## find the shape to which all other samples should be padded to
    #shapes = []

    #for height, width in zip(df.max_height, df.max_width):
    #    shapes.append((height, width))

    #pad_shape = shapes[np.argmax(np.prod(shapes, axis=1))]
    #print("Pad shape", pad_shape)

    # make sure we are not extracting same file again
    assert len(pd.unique(df.fits_fullpath)) == len(df.fits_fullpath)

    pos_files = df.fits_fullpath[df.label==1]
    print("Extracting 7h FITS observation into individual images")
    print("Working on postive samples first")
    print("Files to extract", len(pos_files))
    dest="/data/linn/E5_extracted_pos"
    if not os.path.exists(dest):
        os.mkdir(dest)
    print("Saving to ", dest)
    resize_and_save_in_parallel(pos_files, dest=dest)


    print("Working on negative samples")
    dest="/data/linn/E5_extracted_neg"
    if not os.path.exists(dest):
        os.mkdir(dest)
    neg_files = df.fits_fullpath[df.label==0]
    print("Files to extract", len(neg_files))

    resize_and_save_in_parallel(neg_files, dest=dest)
