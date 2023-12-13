#!/bin/env python
"""
Script for creating two separate text files for positive and negative aarp urls to be downloaded 
It takes the full list of AARP urls as input and assigns labels using the GOES event list catalogue
"""
import pandas as pd
import sys,os
import numpy as np

def split_urllist(df, name):

    urldf = pd.DataFrame({'urls': df[name]})
    urldf[['Datetime', 'AARP', 'Wavelength']] = \
    df[name].str.extract(r'(\d{4}\.\d{2}\.\d{2}_\d{2}:\d{2}:\d{2})_7h@1h_AARP(\d+)_(\d+)\.fits')
    urldf['AARP'] = urldf['AARP'].astype(int)
    urldf['Wavelength'] = urldf['Wavelength'].astype(int)

    return urldf

def split_ondupes(df):

    df = df.assign(NOAA_ARS=df['NOAA_ARS'].str.split(',')).explode('NOAA_ARS').reset_index(drop=True)
    df['NOAA_ARS'] = df['NOAA_ARS'].astype(int)

    return df

def match_noaa_to_harpnum(x):
    match = noaa_to_harps['HARPNUM'].loc[noaa_to_harps['NOAA_ARS']==x].values
    if len(match)==0:
        return -1
    else:
        return match[0]

def remove_offlimb(goes_df):
    # convert string field to tuple
    goes_df['goes_location'] = goes_df.goes_location.apply(eval)
    cutoff_meridian = 70
    print("Removing near limb flares with meridian >", cutoff_meridian)
    return goes_df[np.abs(goes_df.goes_location.apply(lambda x: x[0])) < cutoff_meridian ]

if __name__ == "__main__":
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.abspath(os.path.join(cur_dir,".."))

    aarps_full_df = pd.read_csv(os.path.join(parent_dir,"data/aarps_full_urlist.txt"), header=None, names=['urls'])
    noaa_to_harps = pd.read_csv(os.path.join(parent_dir,"data/all_harps_with_noaa_ars.txt"), delim_whitespace=True)
    orginal_map = noaa_to_harps.copy()
    goes_df = pd.read_csv(os.path.join(parent_dir,"data/GOES_event_list.csv"))
    orginal_goes = goes_df.copy()

    goes_df = remove_offlimb(goes_df)
    urldf = split_urllist(aarps_full_df,'urls')
    noaa_to_harps = split_ondupes(noaa_to_harps)
    flared_harps = pd.unique(noaa_to_harps.HARPNUM)
    urldf['flared_labels'] = urldf.AARP.apply(lambda x: 1 if x in flared_harps else 0)

    goes_df['harpnum'] =  goes_df['noaa_active_region'].apply(match_noaa_to_harpnum)
    goes_df.to_csv("data/GOES_limb_removed.csv", index=False)
    # sub select only AARPS that have resulted in X class flares
    main_class = goes_df['goes_class'].apply(lambda x: x[0])
    urls_x = urldf[urldf['AARP'].isin(goes_df['harpnum'][main_class=='X'])]
    print("Selecting only X class flares")

    n_urls_x = len(urls_x)
    print("Number of X-class flare urls: ", n_urls_x)
    # find number of flares to be selected from negative samples
    grouped =  urls_x.groupby(['AARP','Datetime'])
    x_groups = grouped.ngroups
    print("X-class flare urls grouped by same AARP ID and Timestamp: ", x_groups)

    print("Writing to file -> list of urls for flaring AARPS")
    #urls_x.to_csv(os.path.join(parent_dir,f"data/aarps_pos_{n_urls_x}.csv"), index=False, header=None)

    # select the first n non flaring aarps 
    all_neg = urldf[urldf['flared_labels']==0]
    print("Total number of non-flaring AR observations (urls): ",len(all_neg))
    print("Non-flaring AARPS selected from the top of the list for balanced data: ", n_urls_x)
    balanced_neg = all_neg[:n_urls_x]

    print("Writing to file -> list of urls for non-flaring ARs")
    #balanced_neg.to_csv(os.path.join(parent_dir,f"data/aarps_neg_{n_urls_x}.csv"), index=False, header=None)
