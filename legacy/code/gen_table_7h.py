#!/bin/env python
"""
Script that reads the 7h FITS files and creates a table with metadata
including longitude, image shapes etc to use for selection
"""
import argparse
import pandas as pd
import numpy as np
from astropy.io import fits
from glob import glob
from tqdm import tqdm

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


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-pos-data', '--pos-data')
    parser.add_argument('-neg-data', '--neg-data')
    args = parser.parse_args()

    rows = []

    pos_comp_path = f"{args.pos_data}/*.fits"
    process_dir(pos_comp_path, 1)

    neg_comp_path = f"{args.neg_data}/*.fits"
    process_dir(neg_comp_path, 0)

    table = pd.DataFrame(rows)
    table.to_csv("table_data_shapes.csv")
