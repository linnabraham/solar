import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pandas as pd
from aarp_ml.data_prep import data_prep, gen_table_7h, select_7h, extract_7h, dir_to_json
"""
A sample script which shows how the data processing functions can can be used
"""

if __name__ == "__main__":
    dp = data_prep("./data/GOES_event_list.csv", "./data/all_harps_with_noaa_ars.txt",
                   "./data/aarps_full_urlist.txt")

    # pos_list, neg_list = dp.get_selected_url_df()
    # print(pos_list)

    # table = gen_table_7h()
    table_path = "data/table_data_shapes.csv"
    table = pd.read_csv(table_path, index_col=0)

    # df = select_7h(table)
    selected_7h = "data/selected_7h.csv"
    extracted_dest_pos = "/data/linn/test_extracted_pos"
    extracted_dest_neg = "/data/linn/test_extracted_neg"
    # extract_7h(selected_7h, extracted_dest_pos, extracted_dest_neg)
    extracted_dest_pos = "/data/linn/E6_extracted_pos"
    extracted_dest_neg = "/data/linn/E6_extracted_neg"
    dir_to_json(extracted_dest_pos, extracted_dest_neg)
