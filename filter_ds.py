#!/bin/env python
"""
This script is meant to read the positive and negative AARPS we have selected
and the json file containing the metadata for all the files extracted from these AARPS fits 
files. It selects the central timestamp from the 11 images taken each hour
"""
import os
from astropy.io import fits
import json
from tqdm import tqdm
import logging
logging.basicConfig(level=logging.ERROR)


def ts_from_fits(fits_path):
    """
    Read a single fits file containing 7 hourly observations
    Each of which contains 11 images each
    Return the 7 timestamps of the central image from each of the
    hourly observations
    """
    hdul = fits.open(fits_path)
    header = hdul[0].header
    ts = []
    # iterate over each of the 7 hourly observations
    for hour_num in range(1, header['NTIMES']+1):
        hour_data = hdul[hour_num].data
        try:
            hour_header = hdul[hour_num].header
        except:
            #print("Missing data for hour", hour_num)
            logging.warning(f"Missing data for hour {hour_num}")
        else:
            # get the timestamp of the 6th image
            try:
                timestamp = hour_header["T_IMG05"]
            except:
                #print("Missing timestamp for T_IMG05")
                logging.warning(f"Missing timestamp for T_IMG05")
            else:
                ts.append(timestamp)
    return ts

pos_file_path = "/data/linn/balanced_ds_compressed/pos/"
neg_file_path = "/data/linn/aarps_20231005_3248_neg/"
json_path = "/home/linn/july/solar/solar_dataset_X.json"

with open(json_path) as f:
        data = json.load(f)

pos_files = os.listdir(pos_file_path)
neg_files = os.listdir(neg_file_path)
pos_fullpaths = [os.path.join(pos_file_path,path) for path in pos_files]
neg_fullpaths = [os.path.join(neg_file_path,path) for path in neg_files]

allpaths = []
allpaths.extend(pos_fullpaths)
allpaths.extend(neg_fullpaths)

desired_ts = []

for fits_path in tqdm(allpaths):
    ts = ts_from_fits(fits_path)
    desired_ts.extend(ts)

# avoid iterating over duplicate values
desired_ts = set(desired_ts)
print("Length of desired timestamps", len(desired_ts))

# removing elements from list will mess up indices
# so create a new list to replace the original one
new_training = []

# filter existing timestamps to only select those in the new subset
for timestamp in tqdm(desired_ts, desc="Timestamps"):
    for entry in data['training']:
        # use only first filename as all other wavelengths have same timestamp
        first_filename = entry["0"] 
        if timestamp not in first_filename:
            continue
        else:
            new_training.append(entry)

print(len(new_training))
data['training'] = new_training

with open("solar_dataset.json","w") as write_file:
    json.dump(data, write_file, indent=4)
