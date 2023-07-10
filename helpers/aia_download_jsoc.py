# This script downloads aia data using jsoc api
# Script expects a reference date that is between the t_start and t_end
# The data download method is url-tar which gives us a link by mail
# The script ouputs the response id for the export request

import sunpy.map
from sunpy.net import jsoc
from sunpy.net import attrs as a
import argparse
from astropy.time import Time
import datetime
import astropy.units as u
import time

def time_deltas(ref_date, t_start, t_end):
    del_t1 = datetime.datetime.strptime(ref_date,"%Y-%m-%dT%H:%M:%S.%f")-\
    datetime.datetime.strptime(t_start,"%Y-%m-%dT%H:%M:%S.%f")
    del_t1 = del_t1.total_seconds()

    del_t2 = datetime.datetime.strptime(t_end,"%Y-%m-%dT%H:%M:%S.%f")-\
    datetime.datetime.strptime(ref_date,"%Y-%m-%dT%H:%M:%S.%f")
    del_t2 = del_t2.total_seconds()
    return del_t1, del_t2

def data_download_aia(ref_date:str, del_t1, del_t2, wavelength, fovx, fovy, cadence, email):
    client = jsoc.JSOCClient()
    t = Time(ref_date, format='isot', scale='utc')

    q = client.search(
        a.Time(t - del_t1*u.s, t + del_t2*u.s),
        a.Sample(cadence*u.s),
        a.jsoc.Series('aia.lev1_euv_12s'),
        a.jsoc.Notify(email),
        a.jsoc.Segment("image"),
        a.jsoc.Wavelength(wavelength*u.angstrom)
        #cutout,
        )
    print(q)
    requests = client.request_data(q,method='url-tar')
    print(type(requests))
    print(requests)
    time.sleep(10)
    print(len(q),requests.id,requests.status)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ref_date')
    parser.add_argument('--t_start')
    parser.add_argument('--t_end')
    parser.add_argument('--email')
    parser.add_argument('--fovx', type=int, default=260)
    parser.add_argument('--fovy', type=int, default=260)
    parser.add_argument('--cadence', type=int, default=12)
    parser.add_argument('--wavelength', type=int)

    args = parser.parse_args()

    ref_date = args.ref_date
    t_start = args.t_start
    t_end = args.t_end
    email = args.email
    cadence = args.cadence
    fovx = args.fovx
    fovy = args.fovy
    wavelength = args.wavelength

    del_t1, del_t2 = time_deltas(ref_date, t_start, t_end)
    data_download_aia(ref_date, del_t1, del_t2, wavelength, fovx, fovy, cadence, email)
