# Script that downloads aia data using the SunPy drms module
# A wavelength filter can be applied and the data is downloaded as individual fits files to disk
# An exposure time filter is also harcoded into the query

import argparse
import drms
from pathlib import Path

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--email')
    parser.add_argument('--t_start')
    parser.add_argument('--t_end')
    parser.add_argument('--wavelength', type=int)

    args = parser.parse_args()
    email = args.email
    t_start = args.t_start
    t_end = args.t_end
    wavelength = args.wavelength

    client = drms.Client(email=email)
    keys = ["EXPTIME", "QUALITY", "T_OBS", "T_REC", "WAVELNTH"]

    qstr = f"aia.lev1_euv_12s[{t_start}Z-{t_end}Z][? EXPTIME<2.0 AND WAVELNTH={wavelength} ?]{{image}}"
    print(f"Querying data -> {qstr}")

    records, filenames = client.query(qstr, key=keys, seg="image")
    print(f"{len(records)} records retrieved. \n")
    print(records)
    import sys
    sys.exit(0)

    export = client.export(updated_qstr, method="url", protocol="fits")
    files = export.download(Path("~/sunpy/").expanduser().as_posix())

    #urls = [f"http://jsoc.stanford.edu{filename}" for filename in filenames.image]
    #print(urls)

