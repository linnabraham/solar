# Solar Flare Prediction

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
