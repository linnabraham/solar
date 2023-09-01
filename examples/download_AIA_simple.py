#!/bin/env python
import drms
import os
from time import time

def query_or_download_AIA(t_start, t_end, wavelength: int, email, download=False):
    """
    Function to download AIA images as individual fits files

    Parameters: 
    t_start, t_end: The start and end times of the flare in ISO format
    wavelength: wavelength of observation
    email: JSOC registered email

    Returns:
    Either the names of downloaded files or the records and filename of query result
    """

    client = drms.Client(email=email)
    keys = ["EXPTIME", "QUALITY", "T_OBS", "T_REC", "WAVELNTH"]
    qstr = f"aia.lev1_euv_12s[{t_start}Z-{t_end}Z][? WAVELNTH={wavelength} ?]{{image}}"
    print(f"Querying data -> {qstr}")

    records, filenames = client.query(qstr, key=keys, seg="image")

    if download:
        export = client.export(qstr, method="url", protocol="fits")
        # create a unique dirname using the timestamp of download
        dirname = f"data/{int(time())}"
        os.makedirs(dirname)
        print("Files are downloaded to :", dirname)
        downloaded_files = export.download(dirname)
        return downloaded_files
    else:
        return records, filenames


if __name__=="__main__":    
    flare_start = "2014-01-07T18:04:00"
    flare_end = "2014-01-07T18:58:00"

    #records, filenames = get_AIA(flare_start, flare_end, 94, email, download=False)
    #print(f"{len(records)} records retrieved. \n")
    #print(records)
    email = os.environ['JSOC_EMAIL']
    filenames = get_AIA(flare_start, flare_end, 94, email, download=True)
    print(filenames)

