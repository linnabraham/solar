import argparse
import os
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import tensorflow as tf
import sys
from joblib import delayed
from active_region import active_region
from train_alexnet import get_compiled_model

def make_attribution_movie(filename, data:np.ndarray, channel, timestamps, vmax_frac=0.2, aarp_id=None):
    from matplotlib.animation import FuncAnimation
    nframes = data.shape[0]
    single_channel_mask = data[:,:,:, channel]
    mask_max = np.max(single_channel_mask)
    fig, ax = plt.subplots()
    im = ax.imshow(single_channel_mask[0,:,:], vmax = vmax_frac * mask_max, cmap=plt.cm.jet, origin='lower')

    def update(frame):
        im.set_array(single_channel_mask[frame,:,:])
        if timestamps:
            if aarp_id:
                ax.set_title(f'{timestamps[frame]}_AARP_Id:{aarp_id}_channel_{channel}')
    ani = FuncAnimation(fig, update, frames = nframes, interval=50)
    ani.save(f'{filename}', writer='ffmpeg', fps=1)


def combine_data(goes_event_list, harps_with_noaa):
    goes_df = pd.read_csv(goes_event_list, parse_dates=["event_date", "start_time", "peak_time", "end_time"])
    print("Orginal length of goes_df", len(goes_df))
    harps_with_noaa_df = pd.read_csv(harps_with_noaa, delim_whitespace=True)
    from modified_pipeline import split_onmult, match_noaa_to_harpnum
    harps_with_noaa_df = split_onmult(harps_with_noaa_df)
    goes_df['harpnum'] = goes_df['noaa_active_region'].apply(match_noaa_to_harpnum, args=(harps_with_noaa_df,))
    return goes_df

def add_observations_for_aarp(active_regions_dict, data, aarp_id):
    relevant_entries = [entry for entry in data["test"] if entry["aarp_id"] == aarp_id]
    if aarp_id not in active_regions_dict:
        region = active_region(aarp_id, relevant_entries[0]["label"])  # Assuming all entries have the same label
        active_regions_dict[aarp_id] = region
    else:
        region = active_regions_dict[aarp_id]
    def process_entry(entry):
        timestamp = entry["timestamp"]
        # Add observations for each wavelength from the filtered entries
        for wavelength, fits_path in entry.items():
            if wavelength.isdigit():  # Check if the key is a digit (to exclude "label", "aarp_id", and "timestamp")
                region.add_observation(wavelength, timestamp, fits_path)
    for entry in relevant_entries:
        process_entry(entry)

def single_attribution(model, images, label, args):
    from args_visualize_grads import cross_entropy, get_attributions_mask, plot_attributions_v2
    images = np.where(images<0, np.zeros_like(images), images)
    # create a copy of the images before standardizing for visual plotting
    images_pre = images.copy()
    images = tf.image.per_image_standardization(images)
    attribution_masks  = get_attributions_mask(images, model, target_class_idx=label, args=args)
    images = np.expand_dims(images, axis=0)
    prediction = model.predict(images)

    # compute the cross-entropy loss for the sample
    expected = [ 1.0 - int(label), int(label)]
    predicted = [ 1.0 - prediction, prediction]
    ce = cross_entropy(expected, predicted)
    ce = np.abs(ce)

    return attribution_masks

def single_channel_attribution(model, images):
    # import function for preprocessing E6 data
    from args_visualize_grads import preprocess_data, get_attributions_mask
    images, images_pre = preprocess_data(images)
    print("Shape of images in single_ch_attr function", images.shape)
    attribution_mask = get_attributions_mask(images, model, args)
    return attribution_mask

if __name__=="__main__":
    from tf_utils import get_parser
    parser = get_parser()
    parser.add_argument('--json-path', default="solar_dataset.json")
    parser.add_argument('--trained-model')
    parser.add_argument('--stats-file')
    args = parser.parse_args()

    with open(args.json_path) as json_file:
        data = json.load(json_file)

    ar_dict = {}
    st = time.time()

    aarps_ids_labels = [ (p['aarp_id'], p['label']) for p in data.get('test')]
    aarp_ids = [aarp_id for aarp_id, label in aarps_ids_labels]
    print("Unique AARP_Ids in test", set(aarp_ids))
    add_observations_for_aarp(ar_dict, data, 7304 )


    print(ar_dict.keys())
    first_key, first_value = next(iter(ar_dict.items()))
    print("First element:", first_key)


    ar_data_171, timestamps = first_value.get_observation(171)
    print("AARP Id", first_value.aarp_id)
    ar_data_171 = np.array(ar_data_171)

    active_region.make_aia_movie('crude_movie_171.mp4', ar_data_171, wavelength=171, timestamps = timestamps, aarp_id=first_value.aarp_id, label=first_value.label)


    all_wavelengths = [94,
        131,
        171,
        193,
        211,
        304,
        335]

    model = get_compiled_model(args)
    model.load_weights(args.trained_model)

    alltimes = list(first_value._get_observation_generator(all_wavelengths))
    timestamps = [ timestamp for _, timestamp in alltimes]
    attribution_ts = []
    for multiband_obs,_ in alltimes:
        images = np.array(multiband_obs)
        #attr = single_attribution(model, images, first_value.label, args)
        attr = single_channel_attribution(model, images)
        print("Sum of intensities in attribution:", attr.numpy().sum())
        attribution_ts.append(attr)

    all_attribution_arr = np.array(attribution_ts)
    make_attribution_movie("cool_movie.mp4", all_attribution_arr, channel=1, timestamps = timestamps, aarp_id=first_value.aarp_id)
