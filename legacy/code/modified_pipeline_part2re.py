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

def dir_to_table(fits_dir_path):
    rows = []
    count = 0

    #for fits_file in tqdm(os.listdir(base_data_path_pos)):
    for fits_file in tqdm(os.listdir(fits_dir_path)):
        fits_full_path = os.path.join(fits_dir_path, fits_file)
        hdul = fits.open(fits_full_path)
        main_header = hdul[0].header
        harpnum = main_header['HARPNUM']
        wavelength = main_header['WAVELNTH']
        obs_start = main_header['T_START']

        for hour_num in range(1, main_header['NTIMES']+1):
            data = hdul[hour_num].data
            header = hdul[hour_num].header
            #extname = f"T_IMG{hour_num:0>2d}"

            if data is None:
                print("Empty data encountered in hour number", hour_num, fits_file)
                #continue
            else:
                nimgs = data.shape[0]

                # iterate over 11 images in a single fits extension
                for nimg in range(nimgs):
                    row = {}
                    #print("val of nimg", nimg)
                    #continue
                    img = data[nimg]
                    obstime_key = f"T_IMG{nimg:0>2d}"
                    timestamp = header[obstime_key]
                    #if timestamp == 'NaN':
                    #    print("timstamp missing in header", obstime_key, fits_path)
                    #else:
                    #    row["timestamp"] = timestamp

                    if 'LAT_FWT' in header:
                        lat = header['LAT_FWT']
                    if 'LON_FWT' in header:
                        lon = header['LON_FWT']
                    if 'EXPTIME' in header:
                        exp_time = header['EXPTIME']
                    if 'NOAA_NUM' in header:
                        noaa_match_nos = header['NOAA_NUM']
                    if 'NOAA_AR' in header:
                        noaa_best_match_num = header['NOAA_AR']
                    if 'NOAA_ARS' in header:
                        noaa_arnum_list = header['NOAA_ARS']

                    row["org_file_path"] = fits_file
                    row["harpnum"] = harpnum
                    row["wavelength"] = wavelength
                    row["obs_start"] = obs_start
                    row["org_file_fullpath"] = fits_full_path
                    row["latitude"] = lat
                    row["longitude"] = lon
                    row["noaa_best_match"] = noaa_best_match_num
                    row["hour_num"] = hour_num
                    row["frame_num"] = nimg
                    row["timestamp"] = timestamp
                    row["img_height"] = img.shape[0]
                    row["img_width"] = img.shape[1]

                    rows.append(row)
        count += 1

    table = pd.DataFrame(rows)
    return table

if __name__=="__main__":


    base_data_path_pos = '/data/linn/newpipe_compressed/pos/'
    dest_path = "data/extracted_aarps_pos_details.csv"
    table = dir_to_table(base_data_path_pos)
    print("Length of table", len(table))
    #table.to_csv("test_table.csv", index=False)
    table.to_csv(dest_path, index=False)


    base_data_path_neg = '/data/linn/newpipe_compressed/neg/'
    dest_path = "data/extracted_aarps_neg_details.csv"
    table = dir_to_table(base_data_path_neg)
    table.to_csv(dest_path, index=False)
    #master_df = pd.concat(table_list, ignore_index=True)
    #master_df.set_axis(col_names, axis=1).to_csv(dest_path)

    # Pandas does not produce error if None is present inside the list to concatenate but silently drops
    #for item in table_list:
    #    if item is None:
    #        print("Found!!")
