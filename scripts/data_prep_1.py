from tqdm import tqdm
from aarp_ml.data_prep_new import get_download_list
if __name__ == "__main__":

    # Create list of files to download after applying certain
    # selections
    goes_event_list = "./data/GOES_event_list.csv"
    aarp_full_urls = "./data/aarps_full_urlist.txt"
    harp_to_noaa = "./data/all_harps_with_noaa_ars.txt"

    pos_urls, neg_urls = get_download_list(
            goes_event_list, aarp_full_urls, harp_to_noaa)

    neg_urls.to_csv("neg_samples_df.csv", index=False)
    neg_urls.urls.to_csv("neg_urls.csv", index=False)
    pos_urls.to_csv("pos_samples_df.csv", index=False)
    pos_urls.urls.to_csv("pos_urls.csv", index=False)
