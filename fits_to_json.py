#!/bin/env python
import os, sys
import pandas as pd
import matplotlib.pyplot as plt
import json

pos_data_dir = sys.argv[1]
neg_data_dir = sys.argv[2]

fixed_bands = [94, 131, 171, 193, 211, 304, 335]

def create_dict(data_dir, label):

    filename_list = os.listdir(data_dir)
    fname_df = pd.DataFrame(filename_list, columns=['filename'])
    print("length of filename list dataframe ", len(fname_df))
    # typical filename is like this AARP3556_211_2014-01-03T20:50:25Z.fits
    fname_df[['AARPID','Wavelength','Timestamp']] = fname_df['filename'].str.extract(r'AARP(\d+)_(\d+)_(\d{4}-\d{2}-\d{2}T\d{2}\:\d{2}\:\d{2}Z)\.fits')
    print(fname_df.head())
    fname_df['Wavelength'] = fname_df['Wavelength'].astype(int)
    fname_df['AARPID'] = fname_df['AARPID'].astype(int)
    print("unique length of timestamps", len(pd.unique(fname_df['Timestamp'])))
    obs_ts = []
    training = []
    concat_list = []
    # iterate over a group of images with different wavelengths but same timestamp and aarp id
    grouped =  fname_df.groupby(['AARPID','Timestamp'])
    for group_key, group_df in grouped:
        # convert np.int64 to int for json serialization
        group_aarp_id = int(group_key[0])
        group_timestamp = group_key[1]
        obs_ts.append(len(group_df))
        if len(group_df)==7:
            if set([int(item) for item in group_df['Wavelength'].values]) == set(fixed_bands):
                concat_list.append(group_df)
                training_entry = {
                 "0": os.path.join(data_dir, group_df['filename'].loc[group_df['Wavelength']==fixed_bands[0]].values[0]),
                 "1": os.path.join(data_dir, group_df['filename'].loc[group_df['Wavelength']==fixed_bands[1]].values[0]),
                 "2": os.path.join(data_dir, group_df['filename'].loc[group_df['Wavelength']==fixed_bands[2]].values[0]),
                 "3": os.path.join(data_dir, group_df['filename'].loc[group_df['Wavelength']==fixed_bands[3]].values[0]),
                 "4": os.path.join(data_dir, group_df['filename'].loc[group_df['Wavelength']==fixed_bands[4]].values[0]),
                 "5": os.path.join(data_dir, group_df['filename'].loc[group_df['Wavelength']==fixed_bands[5]].values[0]),
                 "6": os.path.join(data_dir, group_df['filename'].loc[group_df['Wavelength']==fixed_bands[6]].values[0]),
                 "label": label,
                 "aarp_id": group_aarp_id,
                 "timestamp": group_timestamp
                   }
                training.append(training_entry)
    print("No. of timestamps in 7 fixed bands", fixed_bands, " = ",  len(concat_list))
    return training

def write_file(metadata: dict, filename="solar_dataset_X.json"):

    with open(filename, "w") as write_file:
        json.dump(metadata, write_file, indent=4)

if __name__=="__main__":
    #plt.hist(obs_ts)
    #plt.savefig("obs_ts_hist.png")
    trainings = []
    tr_entries = create_dict(pos_data_dir, label="1")
    trainings.extend(tr_entries)
    tr_entries = create_dict(neg_data_dir, label="0")
    trainings.extend(tr_entries)

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
            "training" : trainings
            }

    write_file(metadata)
    pretty = json.dumps(metadata, indent=4)
    #print(pretty)
