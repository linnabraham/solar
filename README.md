# Solar Flare Prediction

## ML Training (2023-10-05)

+ The `examples/scrape_aarps.py` is used to create the full url list from AARPS
+ The `examples/get_negatives_3248.py` is used to create a list of negative samples (balanced with X-class) for download
by external utitilies like `wget`.

## ML Training (2023-09-27)

+ Data used for current training comes from the AARPS database
+ It consists of one FITS file for every day of observation and for every pass band and for every Active Region
+ Each of these FITS files consists of multiple datacubes which span the 7 hr observation
+ The `examples/flare_stats.py` is used to create a list of positive samples (X class flares) to be downloaded with
external utitilies like `wget`.
+ The `examples/make_dataset_single.py` reads files in the directory unpacks the FITS files into individual images 
resizes to a fixed size and saves it as a single FITS file. If there is some data integrity issue the file is skipped and 
the user is notified. The destination to save the extracted files should also be passed to the script
+ The `fits_to_json.py` reads the extracted files and select those for which simultaneous observations exists in the non UV passbands. 
It dumps these filenames as a list into the metadata file named `solar_dataset.json` which is read by the training script.
+ The `train_alexnet.py` file does the actual training. The AlexNet model is implemented in the `helpers/alexnet.py` script.


## Data download

Scripts in the `helpers` directory

+ `get_HMI.py` - Download individual HMI magnetogram patch for an AR (using SunPy.Fido)
+ `get_HMI_single.py` - Class implementation of the same
+ `download_AIA_img.py` - Download AIA image patch based on the HMI patch
+ `aia_download_jsoc.py` - Download a time sequence of images(fits) as a tar (using SunPy.JSOC)
+ `aia_download_drms.py` - Download a time sequence of images(fits) as individual files (using SunPy.drms)
+ `make_movie.py` - Read folder containing fits and process to level 1.5 and create animation and save as mp4

Scripts in the `examples` directory

+ `get_AIA_HMI_patch.py` - Download both HMI and AIA for single patch
+ `download_AIA_example.py` - Download cutout data from AIA based on SunPy example
+ `download_specific_data.py` - AIApy example script for downloading AIA sequence data
