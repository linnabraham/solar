from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import json
from astro_utils import aia
import aarp_ml
from datetime import  timedelta
import matplotlib.gridspec as gridspec
from astro_utils.utils import time_convert, get_start_and_end_time
from aarp_ml.utils import download_GOES_events
from astro_utils.utils import read_fits_single
from aarp_ml.dataset import all_wavelengths
import numpy as np

def read_aarp_fits(fits_filepath):
    with fits.open(fits_filepath) as hdul:
        all_data = []
        hdr = hdul[0].header
        for i in range(1, len(hdul)):  # skip hdul[0], which is empty
            ext_data = hdul[i].data
            if ext_data is not None:
                all_data.append(ext_data.copy())
            else:
                raise ValueError(f"HDU {i} contains no data.")
    return all_data

def plot_aarp_grid(img_arr, passband, vmax_percentile=None, filepath=None, figtitle=None):
    fig, axes = plt.subplots(2, 3, figsize=(10., 4.), constrained_layout=True)

    # Flatten axes array for easy iteration
    axes = axes.flatten()

    for ax, im in zip(axes, img_arr):
        im_obj = aia.plot_aia_image(im, passband=passband, ax=ax, vmax_percentile=vmax_percentile)
    fig.suptitle(figtitle)
        # Add a colorbar using the last image artist
    if im_obj is not None:
        cbar = fig.colorbar(im_obj, ax=axes, orientation='vertical', fraction=0.02, pad=0.04)
        cbar.set_label('Intensity')
    if filepath:
        plt.savefig(filepath, bbox_inches="tight")
    plt.show()

if __name__=="__main__":
    fits_filepath = '/data/linn/E8/compressed/pos/2011.03.09_15:48:00_7h@1h_AARP401_304.fits'
    if os.path.exists(fits_filepath):
        print("yes")
    else:
        print("no")
    data = read_aarp_fits(fits_filepath=fits_filepath)
    max_h = max([d.shape[1] for d in data])
    max_w = max([d.shape[2] for d in data])
    padded_data = np.zeros(shape=(7, 11, max_h, max_w))
    for i, d in enumerate(data):
        _, h, w = d.shape
        padded_data[i, :, :h, :w] = d  # Top-left padding

    movie_data = padded_data.reshape((-1,*padded_data.shape[2:]))
    img_arr = movie_data[::11]
    plot_aarp_grid(img_arr[1:], passband=304, vmax_percentile=99, filepath="aarp_data_grid.png", figtitle="2011.03.09_15:48:00_7h@1h_AARP401_304.fits")
