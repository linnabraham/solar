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
import os,sys
import argparse

def padding(array, xx, yy):
    """
    Function to pad image with zeros to match a target size

    :param array: numpy array
    :param xx: desired height
    :param yy: desired width
    :return: padded array
    """

    h = array.shape[0]
    w = array.shape[1]

    a = (xx - h) // 2
    aa = xx - a - h

    b = (yy - w) // 2
    bb = yy - b - w

    return np.pad(array, pad_width=((a, aa), (b, bb)), mode='constant', constant_values=0)

def downscale(array, xx, yy):
    """
    Downsize an image using scipy interpolation
    """
    from scipy import ndimage
    zfac = (xx/array.shape[0], yy/array.shape[1])
    downscaled = ndimage.zoom(array, zfac, order=1)
    return downscaled

def pad_or_scale(data:np.ndarray):
    """
    Function to decide whether to pad or downscale to achieve fixed size
    based on the original dimensions of the image
    """
    global height
    global width
    orig_height, orig_width = data.shape

    if height < orig_height or width < orig_width:
        image = downscale(data, height, width)
    else:
        image = padding(data, height, width)

    return image



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-input_shape', nargs='+', type=int, default=(512,512))
    args = parser.parse_args()

    height = args.input_shape[0]
    width = args.input_shape[1]

    pos_table_path = "data/extracted_aarps_pos_details.csv"
    neg_table_path = "data/extracted_aarps_neg_details.csv"

    master_df = pd.read_csv(pos_table_path)

    # dont use 1600 for now because the timestamps are different wrt other wavelengths
    master_df = master_df[master_df.wavelength!=1600]
    lon_threshold = 60
    selected_df = master_df[master_df['longitude'].apply(lambda x:np.abs(x)<=lon_threshold)]
    print("No. of rows", len(selected_df))
    print("Unique harps", len(pd.unique(selected_df.harpnum)))
    #concat_df = process(pos_table_path)

    files_to_process = pd.unique(selected_df['org_file_fullpath'])
    print("No. of files to process", len(files_to_process))

    #resize_and_save_in_parallel(files_to_process, dest = "/data/linn/newpipe_extracted_pos")


    base_path = "/data/linn/newpipe_extracted_pos/"

    selected_df["single_img_path"] = selected_df.apply(lambda x: os.path.join(base_path,f"{x[1]}_{x[2]}_{x[10]}.fits"), axis=1)
    selected_df.to_csv("data/pos_traindata_E3.csv", index=False)



    # do for negative
    master_df = pd.read_csv(neg_table_path)

    # dont use 1600 for now because the timestamps are different wrt other wavelengths
    master_df = master_df[master_df.wavelength!=1600]
    lon_threshold = 60
    selected_df = master_df[master_df['longitude'].apply(lambda x:np.abs(x)<=lon_threshold)]
    print("No. of rows", len(selected_df))
    print("Unique harps", len(pd.unique(selected_df.harpnum)))
    #concat_df = process(pos_table_path)

    files_to_process = pd.unique(selected_df['org_file_fullpath'])
    print("No. of files to process", len(files_to_process))

    low_dims=(323,190)
    high_dims=(816, 1444)

    desired_size_df = selected_df[(selected_df.img_height.between(low_dims[0],high_dims[0])) &  (selected_df.img_width.between(low_dims[1], high_dims[1]))]
    print("After filtering with img dimensions")
    print("Rows", len(desired_size_df))
    print("Unique harps", len(pd.unique(desired_size_df.harpnum)))
    files_to_process = pd.unique(desired_size_df['org_file_fullpath'])
    print("No. of files to process", len(files_to_process))


    # no error is raised if this dir doesnt exist

    #resize_and_save_in_parallel(files_to_process, dest = "/data/linn/newpipe_extracted_neg")


    base_path = "/data/linn/newpipe_extracted_neg/"
    desired_size_df["single_img_path"] = desired_size_df.apply(lambda x: os.path.join(base_path,f"{x[1]}_{x[2]}_{x[10]}.fits"), axis=1)
    desired_size_df.to_csv("data/neg_traindata_E3.csv", index=False)
