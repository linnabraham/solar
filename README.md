# Solar Flare Prediction

## Table of Contents

* [AARP Data](#AARP-Data)
* [Quick Start Guide](#Quick-Start-Guide)
* [Advanced](#Advanced)
* [Features Implemented](#Features-Implemented)
* [Legacy Documentation](#Legacy-Documentation)
* [Changelog](#Changelog)

## Data

Links to AARPS

+ The primary dataset can be found here https://umbra.nascom.nasa.gov/contributed/AIA_AARPS/
+ Metadata can be found here https://hpde.io/NASA/NumericalData/SDO/AIA/NWRA/AARP/PT12S
+ Paper that introduces the dataset can be found here https://doi.org/10.3847/1538-4357/ac9c06

## Quick Start Guide

1. Start training

Use the `train_alexnet.py` script.
```
usage: train_alexnet.py [-h] [-input-shape INPUT_SHAPE [INPUT_SHAPE ...]] [-num-channels NUM_CHANNELS] [-json-path JSON_PATH] [-batch-size BATCH_SIZE] [-epochs EPOCHS]
                        [--stats-file STATS_FILE] [--modelpath MODELPATH] [--eval]
```
For example:
```
 python train_alexnet.py --json-path solar_dataset.json --stats-file /data/linn/stats.pkl
```
To use the tensorflow `model.evaluate` for model evaluation on the validation data for sanity check, do the following
```
python train_alexnet.py --json-path solar_dataset.json --stats-file /data/linn/stats.pkl --modelpath outputs/summer-moon-171/best_model.h5 --eval
```
2. Generate Integrated Gradient attribution mask
```
python args_visualize_grads.py --json-path solar_dataset.json --modelpath outputs/summer-moon-171/best_model.h5 --stats-file /data/linn/stats.pkl
```

3. Generate movie from attributions made using consecutive timestamps
```
python new_analysis.py --json-path solar_dataset.json --trained-model outputs/snowy-wildflower-135/best_model.h5
```
4. Debugging the trained model
```
python debug_infer_alexnet.py --json-path solar_dataset.json --modelpath outputs/feasible-leaf-138/model_epoch_05_629.73.weights.h5
```

## Advanced

In the `examples/` directory you can find:
* `scrape_aarps.py` - Scrape all AARP data URLs starting from the data homepage

In the root directory, the following are the most important files:

* `train_alexnet.py` - Script for training alexnet implemented using tensorflow
* `tf_utils.py` - Helper functions used for the training with tensorflow
* `args_visualize_grads.py` - Visualize the output of Integrated Gradients attribution
* `new_analysis.py` - Generate movies using the IG attribution mask for a particular AARP
* `active_region.py` - Defines the active region class, used by the `new_analysis.py` script
* `debug_infer_alexnet.py` - Debug issues with trained model
* `solar_dataset.json` - Training metadata file which is parsed by the training script (not included).
* `stats.pkl` - Pickle file with mean and standard deviation of the data, i.e., training split (not included)

Data pre-processing scripts:

* `modified_pipeline.py` - Create separate list of urls for positive and negative classes
* `gen_table_7h.py` - Create a table with more information derived from FITS headers
* `select_7h.py` - Applies our selection criteria
* `extract_7h.py` - Unpacks 7h FITS files into individual images, pads and saves to disk
* `dir_to_json.py` - Generates json file with metadata to be read during training
* `simple_padding.py` - Helper functions used for padding AARPs to a fixed shape

In the `data/` directory there is:
* `GOES_event_list.csv` - used for labelling AARP into flaring or non-flaring classes
* `aarps_full_urlist.txt` - List of all the AARP URLs
* `all_harps_with_noaa_ars.txt` - Mapping between NOAA numbers and HARPNUM
* `urls_neg_wget.csv` - List of non-flaring AARPs to be downloaded using wget
* `urls_pos_wget.csv` - List of flaring AARPs to be downloaded using wget
* `table_data_shapes.csv` - Table with information about AARPs derived from FITS headers (Not included)
* `selected_7h.csv` - Subset of AARPs after applying selection criteria (Not included)

## Features Implemented

* Use the AARP dataset
* Apply our selection criteria for discarding limb flares, sub X-class etc
* Modified AlexNet architecture with 7 input channels
* Use custom padding scheme to remove sharp discontinuties
* Z-score normalize data along channels
* Apply log transform on the positive and negative pixel values (including those introduced due to z-score normalization)
* Data augmentation using horizontal and vertical flipping
* Interpretable ML techniques such as Integrated Gradients
* Track training using wandb callback
* Data version control using DVC
## Legacy Documentation
### Data download

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

## Changelog
2024-06-05 (Data processing)

+ The scripts used for obtaining and pre-processing the data are `modified_pipeline.py`, `gen_table_7h.py`, `select_7h.py`, `extract_7h.py` and `csv_to_json.py`. 
+ The data pipeline has slightly changed and now we are using `dir_to_json.py` instead of `csv_to_json.py`.
+ The `solar_dataset.json` file is what is finally fed into the network during training.

2023-12-19 (ML Training)

+ We first create a json metadata (`solar_dataset_X.json`) file using the `fits_to_json.py` script. This adds the paths of individual aarp images in 
group of 7 passbands as individual dicts in three lists (training, validation or test). Filtering based on aarp location data is 
done at this stage.
+ To make the dataset more manageable in size we only take the central image from each of the 11 consecutive images taken every hour.
This is done using the `filter_ds.py` script which creates a `solar_dataset.json` file.
+ This is the file that is read by the `train_alexnet.py` script. `evaluate_alexnet.py` script does the evaluation on trained model.
+ The `integ_grad.py` is for model interpretation using Integrated gradient pixel attribution technique.
+ The `helpers/get_location.py` script is used to incorporate location of each 7h aarp by reading the FITS headers and create 
a metadata table with more information that is saved as `aarp_dataframe.csv`

2023-10-05 (ML Training)

+ The `examples/scrape_aarps.py` is used to create the full url list from AARPS
+ The `examples/get_negatives_3248.py` is used to create a list of negative samples (balanced with X-class) for download
by external utitilies like `wget`.

2023-09-27 (ML Training)

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



