import pandas as pd
from glob import glob
import json
import os
import re
from sklearn.model_selection import train_test_split
"""
3. Create a training metadata file as json
"""
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

def dir_to_dataset(dir_path, label):
    """
    Accepts the directory corresponding to positive or negative class and does the processing
    required to generate the json file
    """
    # import pdb; pdb.set_trace()
    fits_full_paths = glob(f"{dir_path}/*.fits")
    df = pd.DataFrame(fits_full_paths, columns=['fits_full_path'])

    df = pd.DataFrame(fits_full_paths, columns=['fits_full_path'])
    df['fits_path'] = df['fits_full_path'].apply(lambda x: os.path.basename(x))

    df[['harpnum','wavelength', 'obs_start', 'timestamp']] = df['fits_path'].apply(split_filepath).apply(pd.Series)
    df['harpnum'] = df['harpnum'].astype(int)
    df['wavelength'] = df['wavelength'].astype(int)

    training = []
    validation = []
    test = []

    aarps_for_train, aarps_for_val, aarps_for_test = split_data(df['harpnum'])
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


def dir_to_json(extracted_dest_pos, extracted_dest_neg):
    training_full = []
    validation_full = []
    test_full = []

    for label, extracted_path in zip((1,0), (extracted_dest_pos, extracted_dest_neg)):
        training, validation, test = dir_to_dataset(extracted_path, label=label)
        training_full.extend(training)
        validation_full.extend(validation)
        test_full.extend(test)

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
            "training" : training_full,
            "validation" : validation_full,
            "test" : test
            }
    pretty = json.dumps(metadata, indent=4)

    filename = "solar_dataset_xx.json"
    with open(filename, "w") as write_file:
        json.dump(metadata, write_file, indent=4)

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


if __name__=="__main__":
    pos_single_dir = "/data/linn/E8/extracted/pos"
    neg_single_dir = "/data/linn/E8/extracted/neg"
    dir_to_json(pos_single_dir, neg_single_dir)
