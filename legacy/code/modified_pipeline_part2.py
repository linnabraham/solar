#!/bin/env python

from astropy.io import fits
import os
import pandas as pd
from tqdm import tqdm

"""
Reads the 7h FITS files in the specified directory and
create a pandas dataframe containing information about all of the individual images 
contained with these FITS and
writes the data to disk as a csv file

The same process is done multiple times for each of two classes.
"""

def fits_to_table(fits_path):
    hdul = fits.open(fits_path)
    main_header = hdul[0].header
    wavelength = main_header['WAVELNTH']
    harpnum = main_header['HARPNUM']
    obs_start = main_header['T_START']
    #images = []
    timestamps = []
    rows = []
    for hour_num in range(1,main_header['NTIMES']+1):
        data = hdul[hour_num].data
        if data is None:
            print("Empty data encountered in hour number", hour_num, fits_path)
            #return None
            continue
        header = hdul[hour_num].header
        #extname = f"T_IMG{hour_num:0>2d}"

        lat = header['LAT_FWT']
        lon = header['LON_FWT']
        exp_time = header['EXPTIME']
        noaa_match_nos = header['NOAA_NUM']
        noaa_best_match_num = header['NOAA_AR']
        noaa_arnum_list = header['NOAA_ARS']

        if lat < 0:
            print("Encountered missing location data for harp num", harpnum)
            #return None
        nimgs = data.shape[0]
        # iterate over 11 images in a single fits extension
        for nimg in range(nimgs):
            img = data[nimg]
            obstime_key = f"T_IMG{nimg:0>2d}"
            timestamp = header[obstime_key]
            if timestamp == 'NaN':
                print("timstamp missing in header", obstime_key, fits_path)
                continue
                #return None
            #images.append(img)
            #timestamps.append(timestamp)

            new_row = [harpnum, 
                       wavelength, 
                       obs_start, 
                       fits_path, 
                       timestamp, 
                       img.shape,
                       lat,
                       lon,
                       exp_time,
                       noaa_match_nos,
                       noaa_arnum_list
                       ]
            rows.append(new_row)
        #table = create_table(harpnum, wavelength, obs_start, fits_path, timestamps, images)
    #rows.append([harpnum, wav, obs_start, filepath, ts, images.shape])
    #for harpnum, wav, obs_start, filepath, ts, images in zip([harpnum]*77, [wavelength]*77, [obs_start]*77, [file_full_path]*77, timestamps, images):
        #rows.append([harpnum, wav, obs_start, filepath, ts, images.shape])
    return pd.DataFrame(rows) 

    
    #return table

#def fits_to_table(fits_path):
#    hdul = fits.open(fits_path)
#    main_header = hdul[0].header
#    wavelength = main_header['WAVELNTH']
#    harpnum = main_header['HARPNUM']
#    obs_start = main_header['T_START']
#    #images = []
#    timestamps = []
#    for hour_num in range(1,main_header['NTIMES']+1):
#        data = hdul[hour_num].data
#        if data is None:
#            print("Empty data encountered in hour number",hour_num, fits_path)
#            return None
#        header = hdul[hour_num].header
#        extname = f"T_IMG{hour_num:0>2d}"
#        nimgs = data.shape[0]
#        # iterate over 11 images in a single fits extension
#        for nimg in range(nimgs):
#            img = data[nimg]
#            obstime_key = f"T_IMG{nimg:0>2d}"
#            timestamp = header[obstime_key]
#            if timestamp == 'NaN':
#                print("timstamp missing in header", obstime_key, fits_path)
#                return None
#            #images.append(img)
#            timestamps.append(timestamp)
#        table = create_table(harpnum, wavelength, obs_start, fits_path, timestamps, images)
#    return table

def create_table(harpnum, wavelength, obs_start, file_full_path, timestamps, images):
    rows = []
    for harpnum, wav, obs_start, filepath, ts, images in zip([harpnum]*77, [wavelength]*77, [obs_start]*77, [file_full_path]*77, timestamps, images):
        rows.append([harpnum, wav, obs_start, filepath, ts, images.shape])
    return pd.DataFrame(rows) 

def dir_to_table(base_data_path):
    table_list = []
    for fits_file in tqdm(os.listdir(base_data_path)):
        table = fits_to_table(os.path.join(base_data_path, fits_file))
        if table is not None:
            table = table.reset_index()
            table = table.rename(columns={'index':'old_index'})
            table_list.append(table)
    return table_list

col_names = ["old_index", "harpnum", "wavelength", "obs_start", "file_full_path", "timestamp", "dimensions", "Latitude", "Longitude", "Exposure","NOAA_nmatches", "NOAA_AR_nums"]

base_data_path_pos = '/data/linn/newpipe_compressed/pos/'
dest_path = "data/extracted_aarps_pos_details.csv"
table_list = dir_to_table(base_data_path_pos)
master_df = pd.concat(table_list, ignore_index=True)
master_df.set_axis(col_names, axis=1).to_csv(dest_path)

base_data_path_neg = '/data/linn/newpipe_compressed/neg/'
dest_path = "data/extracted_aarps_neg_details.csv"
table_list = dir_to_table(base_data_path_neg)
master_df = pd.concat(table_list, ignore_index=True)
master_df.set_axis(col_names, axis=1).to_csv(dest_path)

# Pandas does not produce error if None is present inside the list to concatenate but silently drops
#for item in table_list:
#    if item is None:
#        print("Found!!")
