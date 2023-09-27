#!/bin/env python
"""
Script for iterating over the individual AARPS fits files present in a directory 
Unpacking the 7 cubes each containing 11 images 
Pad or downscale each image to make into a fixed size (downscale not preserving aspect ratio)
Save to disk as individual fits files with timestamp and wavelength in the filenames
"""
import os,sys
sys.path.append("../")
import numpy as np
from astropy.io import fits

def padding(array, xx, yy):
    """
    Function to pad image with zeros to match a target size

    :param array: numpy array
    :param xx: desired height
    :param yy: desired width
    :return: padded array
    """

    h = array.shape[0]
    w = array.shape[1]

    a = (xx - h) // 2
    aa = xx - a - h

    b = (yy - w) // 2
    bb = yy - b - w

    return np.pad(array, pad_width=((a, aa), (b, bb)), mode='constant', constant_values=0)

def downscale(array, xx, yy):
    """
    Downsize an image using scipy interpolation
    """
    from scipy import ndimage
    zfac = (xx/array.shape[0], yy/array.shape[1])
    downscaled = ndimage.zoom(img, zfac, order=1)
    return downscaled

def read_fits(file_path):
    hdul = fits.open(file_path)
    return hdul

def pad_or_scale(data:np.ndarray, height, width):
    """
    Function to decide whether to pad or downscale to achieve fixed size
    based on the original dimensions of the image
    """
    orig_height, orig_width = data.shape

    if height < orig_height or width < orig_width:
        image = downscale(data, height, width)
    else:
        image = padding(data, height, width)

    return image


if __name__=="__main__":

    flare_start = "2014-01-07T18:04:00"
    flare_end = "2014-01-07T18:58:00"

    aarp_data_path = sys.argv[1]
    dest = sys.argv[2]
    counter = 0
    # iterate over individual fits file in folder
    for file_name in os.listdir(aarp_data_path):
        file_path = os.path.join(aarp_data_path,file_name)
        hdul = read_fits(file_path)
        header = hdul[0].header
        wavelength = header['WAVELNTH']
        # iterate over the 7 channels in single fits
        for channel_num in range(1, header['NTIMES']+1):
            data = hdul[channel_num].data
            if data is None:
                print("Empty data encountered in ", file_name)
                continue
            header = hdul[channel_num].header
            extname = f"T_IMG{channel_num:0>2d}"
            nimgs = data.shape[0]
            # iterate over the 11 images in a single fits extension or channel
            for nimg in range(nimgs):
                img = data[nimg]
                obstime_key = f"T_IMG{nimg:0>2d}"
                timestamp = header[obstime_key]
                new_img = pad_or_scale(img, 512, 512)
                hdu = fits.PrimaryHDU(data=new_img)
                new_hdul = fits.HDUList([hdu])
                file_name_noext = os.path.splitext(file_name)[0]
                aarp_id, wavelength = file_name_noext.split("_")[3:5]
                fits_filename = f'{aarp_id}_{wavelength}_{timestamp}.fits'
                if dest:
                    new_hdul.writeto(os.path.join(dest,fits_filename))
                counter+=1

