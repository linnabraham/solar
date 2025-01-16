import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pandas as pd
from aarp_ml.data_prep import data_prep, gen_table_7h, select_7h, extract_7h, dir_to_json
"""
A sample script which shows how the data processing functions can can be used
"""

if __name__ == "__main__":
    goes_event_list = "./data/GOES_event_list.csv"
    harp_to_noaa = "./data/all_harps_with_noaa_ars.txt"
    aarp_full_urls = "./data/aarps_full_urlist.txt"

    dp = data_prep(goes_event_list, harp_to_noaa, aarp_full_urls)

    goes_df = dp.goes_df
    aarps_full_df = dp.aarps_full_df
    harps_with_noaa_df = dp.harps_with_noaa_df

    aarps_clean_df = dp.clean_url_df(aarps_full_df)
    goes_clean_df = dp.get_clean_goes_df(goes_df,  harps_with_noaa_df, flare_class="X")
    url_df = dp.get_selected_url_df(aarps_clean_df,  goes_clean_df)

    pos_dir_7h = "/data/linn/newpipe_compressed/pos"
    neg_dir_7h = "/data/linn/newpipe_compressed/neg"
    table = gen_table_7h(pos_dir_7h, neg_dir_7h)
    # table_path = "data/table_data_shapes.csv"
    table = pd.read_csv(table_path, index_col=0)
    selected_df = select_7h(table)
    # selected_7h = "data/selected_7h.csv"

    pos_dir_single = "/data/linn/E6_extracted_pos"
    neg_dir_single = "/data/linn/E6_extracted_neg"

    # writes files to disk
    extract_7h(selected_df, pos_dir_single, neg_dir_single)

    # saves to solar_dataset_xx.json
    dir_to_json(pos_dir_single, neg_dir_single)
