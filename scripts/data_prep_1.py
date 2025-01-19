from tqdm import tqdm
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
np.random.seed(42)
"""Scripts used for data download and processing
without using any class functions
1. Create list of files to download after applying certain
selections
"""
def select_urls(urldf, goes_df):
    urldf_copy = urldf.copy()
    urldf_copy['goes_matched_start'] = None
    for index, row in tqdm(urldf.iterrows(), total=len(urldf)):
        aarp_id = row['AARP']
        obs_start = datetime.strptime(row['Datetime'], "%Y.%m.%d_%H:%M:%S")
        matching_rows = goes_df[ (goes_df['harpnum'] == aarp_id) & (obs_start + timedelta(hours=6) > goes_df['start_time'][goes_df['harpnum'] == aarp_id])]
        if any(matching_rows):
            if not matching_rows.empty:
                matched_start_time = matching_rows['start_time'].values[0]
                urldf_copy.at[index, 'goes_matched_start'] = matched_start_time
    return urldf_copy

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

def label_urls(urldf, goes_df):
        urldf['label'] = -99
        flared_aarp_ids = set(goes_df.harpnum)
        for index, row in tqdm(urldf.iterrows(), total=len(urldf)):
            aarp_id = row['AARP']
            if not any(goes_df['harpnum'] == aarp_id):
                urldf.at[index, 'label'] = 0
            elif any(goes_df['harpnum'] == aarp_id):
                urldf.at[index, 'label'] = 1
        return urldf

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

def select_neg_urls(urldf, num_aarps):
    urldf = urldf.sample(frac=1).reset_index(drop=True)
    neg_aarp_ids = urldf.AARP.unique()[:num_aarps]
    urldf = urldf[urldf.AARP.isin(neg_aarp_ids)]

    group_keys = list(urldf.groupby(["Datetime", "AARP"]).groups.keys())
    np.random.shuffle(group_keys)
    collected_groups = []

    for key in group_keys:
        df = urldf.groupby(["Datetime", "AARP"]).get_group(key)
        collected_groups.append(df)

    urldf = pd.concat(collected_groups, ignore_index=True)
    return urldf

if __name__ == "__main__":
    goes_event_list = "./data/GOES_event_list.csv"
    goes_df = pd.read_csv(goes_event_list, parse_dates=["event_date", "start_time", "peak_time", "end_time"])

    aarp_full_urls = "./data/aarps_full_urlist.txt"
    aarps_full_df = pd.read_csv(aarp_full_urls, header=None, names=['urls'])
    aarps_clean_df = split_urllist(aarps_full_df, "urls")
    aarps_clean_df = aarps_clean_df[aarps_clean_df["Wavelength"] != 1600]

    harp_to_noaa = "./data/all_harps_with_noaa_ars.txt"
    harp_to_noaa_df =pd.read_csv(harp_to_noaa, delim_whitespace=True)
    harp_to_noaa_df = split_onmult(harp_to_noaa_df)

    goes_df['harpnum'] = goes_df.noaa_active_region.apply(match_noaa_to_harpnum, args=(harp_to_noaa_df,))
    goes_df = goes_df[goes_df.harpnum != -1]

    urldf = label_urls(aarps_clean_df, goes_df)
    pos_urls = urldf[urldf.label==1]

    selected_urls = select_urls(pos_urls, goes_df)

    matched_urls = selected_urls[selected_urls.goes_matched_start.notna()]
    print(f"URLs with matches:", len(matched_urls))

    pos_urls = selected_urls[~selected_urls.goes_matched_start.notna()]
    print(f"URLs in positive class:", len(pos_urls))

    neg_urls = urldf[urldf.label==0]
    num_aarps = pos_urls.AARP.nunique()*4
    neg_urls = select_neg_urls(neg_urls, num_aarps)
    print(f"URLs in negative class:", len(matched_urls))

    neg_urls.to_csv("neg_samples_df.csv", index=False)
    neg_urls.urls.to_csv("neg_urls.csv", index=False)
    pos_urls.to_csv("pos_samples_df.csv", index=False)
    pos_urls.urls.to_csv("pos_urls.csv", index=False)
