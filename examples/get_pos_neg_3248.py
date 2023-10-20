#!/bin/env python
import pandas as pd
import sys,os

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

if __name__=="__main__":
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.abspath(os.path.join(cur_dir,".."))
    aarps_full_df = pd.read_csv(os.path.join(parent_dir,"data/aarps_full_urlist.txt"), header=None, names=['urls'])
    urldf = split_urllist(aarps_full_df,'urls')
    noaa_to_harps = pd.read_csv(os.path.join(parent_dir,"data/all_harps_with_noaa_ars.txt"), delim_whitespace=True)
    orginal_map = noaa_to_harps.copy()
    noaa_to_harps = split_ondupes(noaa_to_harps)
    flared_harps = pd.unique(noaa_to_harps.HARPNUM)
    goes_df = pd.read_csv(os.path.join(parent_dir,"data/GOES_event_list.csv"))
    orginal_goes = goes_df.copy()
    goes_df['harpnum'] =  goes_df['noaa_active_region'].apply(match_noaa_to_harpnum)
    urldf['flared_labels'] = urldf.AARP.apply(lambda x: 1 if x in flared_harps else 0)

    main_class = goes_df['goes_class'].apply(lambda x: x[0])
    urls_x = urldf[urldf['AARP'].isin(goes_df['harpnum'][main_class=='X'])]
    print("Writing to file -> list of urls for  all X-class flares")
    urls_x.to_csv(os.path.join(parent_dir,f"data/aarps_pos_{n_urls_x}.csv"), index=False, header=None)

    grouped =  urls_x.groupby(['AARP','Datetime'])
    x_groups = grouped.ngroups
    n_urls_x = len(urls_x)
    print("Number of X-class flare urls", n_urls_x)
    print("X-class flare urls grouped by Wavelength", x_groups)
    all_neg = urldf[urldf['flared_labels']==0]
    print("Number of non-flaring AR observations (urls)",len(all_neg))
    balanced_neg = all_neg[:n_urls_x]
    print("Writing to file -> list of urls for non-flaring ARs (balanced)")
    #balanced_neg.urls.to_csv(os.path.join(parent_dir,f"data/aarps_neg_{n_urls_x}.csv"), index=False, header=None)
