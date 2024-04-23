#!/bin/env python
"""
Script for generating the json metadata file that containing image paths to be used for
training, validation and testing. The information in the file paths are used for this.
We make sure the AARPS in each set are unique
"""
import argparse
from glob import glob
import pandas as pd
import re
import os
import json
from modified_pipeline import split_urllist
from csv_to_json import filter_central_ts, split_data, simultaneous_multiband

def split_filepath(file_path):
    """
    Function that reads the name of individual file as a string and extracts AARP id, wavelength, observation
    start time and the timestamp encoded in the string.
    """
    #assert '/' not in file_path
    # 3364_171_2013.11.12_15:48:00_TAI_2013-11-12T21:53:49Z.fits
    pattern = r'(\d+)_(\d+)_(\d{4}\.\d{2}\.\d{2}_\d{2}:\d{2}:\d{2})_TAI_(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z)'
    match = re.search(pattern, file_path)
    if match:
        result = match.groups()
        return(result)
    else:
        print("No match found.")
        return None

def dir_to_dataset(dir_path, label):
    """
    Accepts the directory corresponding to positive or negative class and does the processing
    required to generate the json file
    """
    fits_full_paths = glob(f"{dir_path}/*.fits")
    df = pd.DataFrame(fits_full_paths, columns=['fits_full_path'])

    df = pd.DataFrame(fits_full_paths, columns=['fits_full_path'])
    df['fits_path'] = df['fits_full_path'].apply(lambda x: os.path.basename(x))

    df[['harpnum','wavelength', 'obs_start', 'timestamp']] = df['fits_path'].apply(split_filepath).apply(pd.Series)
    df['harpnum'] = df['harpnum'].astype(int)
    df['wavelength'] = df['wavelength'].astype(int)

    central_ts = filter_central_ts(df)

    training = []
    validation = []
    test = []

    aarps_for_train, aarps_for_val, aarps_for_test = split_data(central_ts['harpnum'])
    print(len(aarps_for_train), len(aarps_for_val), len(aarps_for_test))

    fixed_bands = [94, 131, 171, 193, 211, 304, 335]

    for group_harpnum, group_timestamp, group_df in simultaneous_multiband(df):

        result =  group_df.loc[group_df['wavelength']==fixed_bands[0]].values[0]

        entry = {
         "0": group_df['fits_full_path'].loc[group_df['wavelength']==fixed_bands[0]].values[0],
         "1": group_df['fits_full_path'].loc[group_df['wavelength']==fixed_bands[1]].values[0],
         "2": group_df['fits_full_path'].loc[group_df['wavelength']==fixed_bands[2]].values[0],
         "3": group_df['fits_full_path'].loc[group_df['wavelength']==fixed_bands[3]].values[0],
         "4": group_df['fits_full_path'].loc[group_df['wavelength']==fixed_bands[4]].values[0],
         "5": group_df['fits_full_path'].loc[group_df['wavelength']==fixed_bands[5]].values[0],
         "6": group_df['fits_full_path'].loc[group_df['wavelength']==fixed_bands[6]].values[0],
         "label": label,
         "aarp_id": group_harpnum,
         "timestamp": group_timestamp
           }

        if group_harpnum in aarps_for_train:
            training.append(entry)
        elif group_harpnum in aarps_for_val:
            validation.append(entry)
        elif group_harpnum in aarps_for_test:
            test.append(entry)
        else:
            print("Found aarp id not in given list")

    return training, validation, test

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-pos-data", "--pos-data")
    parser.add_argument("-neg-data", "--neg-data")
    args = parser.parse_args()

    pos_dir_path = args.pos_data

    pos_training, pos_validation, pos_test = dir_to_dataset(pos_dir_path, label=1)

    neg_dir_path = args.neg_data
    neg_training, neg_validation, neg_test = dir_to_dataset(neg_dir_path, label=0)

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
            "training" : pos_training + neg_training,
            "validation" : pos_validation + neg_validation,
            "test" : pos_test + neg_test
            }
    pretty = json.dumps(metadata, indent=4)

    filename = "solar_dataset.json"
    with open(filename, "w") as write_file:
        json.dump(metadata, write_file, indent=4)

