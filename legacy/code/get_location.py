import os, sys
from glob import glob
from astropy.io import fits
from sunpy.map import Map
import pandas as pd

"""
This script receives as inputs the location of the 7hour combined AARP data
for both positive and negative classes.
It creates a table with one row for each AARP containing metadata such 
as patch dimensions, locations, wavelength etc.
We need to iterate over individual hourly fits to get such information from 
the header but since these are same for all the hourly observations we are
breaking the iteration after the first one.
"""
if __name__=="__main__":
    aarp_pos_path = sys.argv[1]
    aarp_neg_path = sys.argv[2]
    file_names = os.listdir(aarp_pos_path)
    file_names = glob(f"{aarp_pos_path}/*.fits")
    file_names.extend(glob(f"{aarp_neg_path}/*.fits"))
    print(len(file_names))
    rows = []
    # iterate over individual fits file in folder
    for file_name in file_names:
        if 'pos' in file_name:
            label = 1
        elif 'neg' in file_name:
            label = 0
        else:
            print("ambigous label")
        size = os.path.getsize(file_name)
        size_in_MB = size/(1024*1024)

        #file_path = os.path.join(aarp_data_path,file_name)
        hdul = fits.open(file_name)
        #print(hdul[0].header)
        # There is metadata in index 0 and 1 - 7 contains the 7 hourly data cubes
        for hour_num in range(1, hdul[0].header['NTIMES']+1):
            data = hdul[hour_num].data
            if data is None:
                print("Empty data encountered in ", file_name)
                continue
            header = hdul[hour_num].header
            #print(dir(header))
            #print([key for key in header.keys()])
            # selected keys
            nimgs = data.shape[0]
            #print(data.shape)
            x_coord = header['NAXIS1']
            y_coord = header['NAXIS2']
            harp_num = header['HARPNUM']
            noaa_match_nos = header['NOAA_NUM']
            noaa_best_match_num = header['NOAA_AR']
            noaa_arnum_list = header['NOAA_ARS']
            lat = header['LAT_FWT']
            if lat < 0:
                print("Encountered missing location data for harp num", harp_num)
            lon = header['LON_FWT']
            exp_time = header['EXPTIME']
            wavelength = header['WAVELNTH']
            #time = header[f"T_IMG{nimgs:0>2d}"]
            start_time = header['T_IMG00']
            end_time = header['T_IMG10']
            row = {"x_coord":x_coord,
                    "y_coord":y_coord,
                    "harp_num":harp_num,
                    "noaa_match_nos":noaa_match_nos,
                    "noaa_arnum_list":noaa_arnum_list,
                    "lat":lat,
                    "lon":lon,
                    "exposure_time":exp_time,
                    "wavelength":wavelength,
                    "start_time":start_time,
                    "end_time":end_time,
                    "size_MB" : size_in_MB,
                    "label" : label
                    }
            rows.append(row)
            #smap = Map(data, header)
            #print(smap.wcs)
            #smap.plot()
            break

        #break
    df = pd.DataFrame(rows)
    #print(df)
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    # define the base path to be 1 level up from the location of file
    base_path = os.path.abspath(os.path.join(cur_dir,".."))
    df.to_csv(os.path.join(base_path,"aarp_dataframe.csv"),index=False)

