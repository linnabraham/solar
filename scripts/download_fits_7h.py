from pathlib import Path
import os, sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
from helpers.fits_parallel_download import download_urls_in_parallel

if __name__ == "__main__":
    output_dir = Path("/data/linn/E8/compressed/pos/")
    # output_dir = Path("/data/linn/E8/compressed/neg/")
    output_dir.mkdir(exist_ok=True)

    csv_file = os.path.join(parent_dir, "pos_urls.csv")
    # csv_file = "neg_urls.csv"
    download_urls_in_parallel(csv_file, output_dir, max_workers=10)
