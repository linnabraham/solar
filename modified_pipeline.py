#!/bin/env python

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from datetime import timedelta
import os, sys
from tqdm import tqdm

def split_onmult(df):
    """
    If there are multiple harpnums matching a single NOAA_ARS number turn those into extra rows
    """
    df = df.assign(NOAA_ARS=df['NOAA_ARS'].str.split(',')).explode('NOAA_ARS').reset_index(drop=True)
    df['NOAA_ARS'] = df['NOAA_ARS'].astype(int)

    return df

def match_noaa_to_harpnum(x, harps_with_noaa_df):
    """
    Match each noaa number to a harpnum
    If not found return -1
    """
    match = harps_with_noaa_df['HARPNUM'].loc[harps_with_noaa_df['NOAA_ARS']==x].values
    if len(match)==0:
        return -1
    else:
        return match[0]

def split_urllist(df, name):
    """
    Using regex matching convert the url paths into seperate columns
    """
    urldf = pd.DataFrame({'urls': df[name]})
    urldf[['Datetime', 'AARP', 'Wavelength']] = \
    df[name].str.extract(r'(\d{4}\.\d{2}\.\d{2}_\d{2}:\d{2}:\d{2})_7h@1h_AARP(\d+)_(\d+)\.fits')
    urldf['AARP'] = urldf['AARP'].astype(int)
    urldf['Wavelength'] = urldf['Wavelength'].astype(int)

    return urldf

data_dir = "./data"
goes_df = pd.read_csv(os.path.join(data_dir,"GOES_event_list.csv"), parse_dates=["event_date", "start_time", "peak_time", "end_time"])
harps_with_noaa_df = pd.read_csv(os.path.join(data_dir,"all_harps_with_noaa_ars.txt"), delim_whitespace=True)
aarps_full_df = pd.read_csv(os.path.join(data_dir,"aarps_full_urlist.txt"), header=None, names=['urls'])

harps_with_noaa_df = split_onmult(harps_with_noaa_df)
goes_df['harpnum'] = goes_df['noaa_active_region'].apply(match_noaa_to_harpnum, args=(harps_with_noaa_df,))
goes_df_org = goes_df.copy()

# remove cases where there is no corresponding noaa AR number that matches
goes_df  = goes_df.query("harpnum != -1")
# remove flares reported by goes that are off-limb
#goes_df = goes_df[goes_df['goes_location'].apply(lambda x: np.abs(eval (x)[1]) < 60)]
# select only AARPS that have resulted in major flares
goes_df = goes_df[goes_df['goes_class'].apply(lambda x: x[0]) == "X"]

urldf = split_urllist(aarps_full_df, "urls")
urldf['label'] = -99
urldf['goes_matched_start'] = None
#for index, row in tqdm(urldf[:50000].iterrows(), total=len(urldf[:50000])):
for index, row in tqdm(urldf.iterrows(), total=len(urldf)):
    obs_start = datetime.strptime(row['Datetime'], "%Y.%m.%d_%H:%M:%S")
    aarp_id = row['AARP']
    #if aarp_id in goes_df_org.harpnum:
        #print("yes")
    #if any((goes_df_org['harpnum'] == aarp_id) & (obs_start + timedelta(hours=7) < goes_df_org['start_time'])):
    #    urldf.at[index, 'label'] = 1
    #    urldf.at[index, 'goes_flare_start'] = 1
    
    # if the aarp id does not belong to one that has ever flared according to goes label it 0
    if not any(goes_df_org['harpnum'] == aarp_id):
        urldf.at[index, 'label'] = 0

    #elif any((goes_df_org['harpnum'] == aarp_id) & (obs_start + timedelta(hours=7) < goes_df_org['start_time'])):
    elif any((goes_df['harpnum'] == aarp_id) & (obs_start + timedelta(hours=7) < goes_df['start_time'])):
        matching_rows = goes_df[(goes_df['harpnum'] == aarp_id) & (obs_start + timedelta(hours=7) < goes_df['start_time'])]
        #matching_rows = goes_df_org[(goes_df_org['harpnum'] == aarp_id) & (obs_start + timedelta(hours=7) < goes_df_org['start_time'])]

        if not matching_rows.empty:
            matched_start_time = matching_rows['start_time'].values[0]
            urldf.at[index, 'goes_matched_start'] = matched_start_time
            urldf.at[index, 'label'] = 1

print("No. of 7hr AARP observation matches", (urldf['label']==1).sum())
#print("No. of 7hr AARP observation with linked goes flare", (urldf['goes_matched_start'] is not None).sum())
print("No. of unique AARPS", pd.unique(urldf[urldf['label']==1].AARP))

print(urldf[urldf['goes_matched_start'].notna()])
#print(urldf.head())
#urldf.to_csv("data/urls_to_download.csv", index=False)
#urldf['urls'][urldf['label']==1].to_csv("data/urls_pos_wget.csv", index=False, header=None)
#TODO:incorporate grouping by wavelength and then shuffling to select the negative AARPS
urldf['urls'][urldf['label']==0][:5000].to_csv("data/urls_neg_wget.csv", index=False, header=None)
