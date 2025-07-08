# Solar Flare Prediction

## Table of Contents
* [Data](#Data)
* [Usage](#Usage)
## Data

Links to AARPS

+ The primary dataset can be found here https://umbra.nascom.nasa.gov/contributed/AIA_AARPS/
+ Metadata can be found here https://hpde.io/NASA/NumericalData/SDO/AIA/NWRA/AARP/PT12S
+ Paper that introduces the dataset can be found here https://doi.org/10.3847/1538-4357/ac9c06

## Usage
All scripts have to be run with `python -m name-of-script`

### Setup Environment
```
conda install -c conda-forge mamba # Use `mamba` instead of `conda`
conda create -n torch-tf-312
conda env update --file torch-tf-312.environment.yml
```

### Setup DVC

### Download and Pre-process data



- `src.data_single --download --process --select --extract --json --stats`
    - This script downloads the 6 hour AARP data, applies the selection criteria, extracts individual images and does the padding and stores it in a different directory, does the train/val/test split and creates a config file (json) to be passed to the training script and also computes the mean and standard deviations on the training split for use in z-score normalization.


### Training:

- `src.torch.xgb_train` - Train xgboost model
- `src.torch.train_alexnet` - Train AlexNet model
- `src.torch.vit.train` - Train ViT model

### Analysis:
- `src.torch.vit.class_wise_distribution` - Create IG attributions and save the data to disk 
- `src.torch.vit.plot_class_wise_distribution` - Plot the classwise distributions using the attributions read from the saved file.
- `src.torch.vit.predictions_analyze` - Analyze the predictions of the ViT model for each AARP sequence in the val/test set.
- `src.torch.xgb_feat_importance.py` - Plot the feature importance for the xgboost model.

### Misc
- `src.check_json` - Check the json config file for filepaths that do not exist on disk.
- `src.clean_json` - Remove any missing file paths from config and write it to a new file.
