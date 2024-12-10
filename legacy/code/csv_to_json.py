#!/bin/env python
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import json

def split_data(harpnums: pd.Series):
    """
    Given a list of harp ids find the unique ids and split it into train, val and test
    so that no two sets have the same harp id
    """
    unique_harps = pd.unique(harpnums)
    train_data, test_data = train_test_split(unique_harps, test_size=0.2, random_state=42)
    train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)
    aarp_lists = (train_data, val_data, test_data)
    return aarp_lists

def simultaneous_multiband(df):
    """
    Accepts a dataframe with each row referring to a single AARP observation
    Return a generator that generates groups of observations
    with the same timestamp and harp id but different wavelengths
    """
    fixed_bands = [94, 131, 171, 193, 211, 304, 335]
    grouped = df.groupby(['harpnum','timestamp'])
    for group_key, group_df in grouped:
        #assert len(group_df) ==7
        if len(group_df) == 7:
            assert set([int(item) for item in group_df['wavelength'].values]) == set(fixed_bands)
            group_harpnum = int(group_key[0])
            group_timestamp = group_key[1]
            yield (group_harpnum, group_timestamp, group_df)

def filter_central_ts(df):
    concat_list = []
    grouped = df.groupby(['obs_start','wavelength'])
    for group_key, group_df in grouped:
        central_ts = group_df.iloc[0::11]
        #print(len(central_ts))
        #assert len(central_ts) == 77
        if central_ts is not None:
            concat_list.append(central_ts)
        else:
            print(f"Got none for {group_key}")
    concat_df = pd.concat(concat_list)
    return concat_df
    
def create_data(csv_path, label):
    df = pd.read_csv(csv_path)
    df = filter_central_ts(df)
    fixed_bands = [94, 131, 171, 193, 211, 304, 335]

    training = []
    validation = []
    test = []

    aarps_for_train, aarps_for_val, aarps_for_test = split_data(df['harpnum'])
    print(len(aarps_for_train), len(aarps_for_val), len(aarps_for_test))

    for group_harpnum, group_timestamp, group_df in simultaneous_multiband(df):
        assert group_harpnum is not None
        assert group_timestamp is not None
        #print(group_harpnum)
        #print(group_df)
        #result =  group_df.loc[group_df['wavelength']==fixed_bands[0]]
        #print(result)
        entry = {
         "0": group_df['single_img_path'].loc[group_df['wavelength']==fixed_bands[0]].values[0],
         "1": group_df['single_img_path'].loc[group_df['wavelength']==fixed_bands[1]].values[0],
         "2": group_df['single_img_path'].loc[group_df['wavelength']==fixed_bands[2]].values[0],
         "3": group_df['single_img_path'].loc[group_df['wavelength']==fixed_bands[3]].values[0],
         "4": group_df['single_img_path'].loc[group_df['wavelength']==fixed_bands[4]].values[0],
         "5": group_df['single_img_path'].loc[group_df['wavelength']==fixed_bands[5]].values[0],
         "6": group_df['single_img_path'].loc[group_df['wavelength']==fixed_bands[6]].values[0],
         "label": label,
         "aarp_id": group_harpnum,
         "timestamp": group_timestamp
           }

        #print(entry)
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
    csv_path = "data/pos_traindata_E3.csv"

    fixed_bands = [94, 131, 171, 193, 211, 304, 335]

    pos_training, pos_validation, pos_test = create_data(csv_path, label=1)
    print(len(pos_training), len(pos_validation), len(pos_test))

    csv_path = "data/neg_traindata_E3.csv"

    neg_training, neg_validation, neg_test = create_data(csv_path, label=0)
    print(len(neg_training), len(neg_validation), len(neg_test))

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
    #print(pretty)
    filename = "solar_dataset.json"
    with open(filename, "w") as write_file:
        json.dump(metadata, write_file, indent=4)

