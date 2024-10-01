import argparse
import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import backend
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import tempfile
import matplotlib
from active_region import active_region
from train_alexnet import get_compiled_model
from args_visualize_grads import preprocess_data, get_attributions_mask

def attribution_contour(filename, raw:np.ndarray, attribs:np.ndarray, wavelength, 
                        channel, timestamps, vmax_frac=0.2, aarp_id=None):
    """
    raw: Time sequence of raw AARP observations
    attribs: Time sequence of IG attributions made using raw AARP observations
    wavelength: AIA passband for display
    channel: The channel index corresponding to the passband
    timestamps: List containing timestamps
    """
    nframes = raw.shape[0]
    cmap_key = 'sdoaia'+str(wavelength)
    sdoaia_cmap = matplotlib.colormaps[cmap_key]

    single_channel_mask = attribs[:,:,:, channel]
    mask_max = np.max(single_channel_mask)
    fig, ax = plt.subplots()
    data = np.where(raw < 0, np.zeros_like(raw), raw)
    im = ax.imshow(np.sqrt(data[0,:,:]), cmap=sdoaia_cmap, origin='lower')
    contour = None

    def update(frame):
        nonlocal contour
        im.set_array(np.sqrt(data[frame,:,:]))
        if contour is not None:
            for c in contour.collections:
                c.remove()
        contour = ax.contour(single_channel_mask[frame, :, :], vmax= vmax_frac * mask_max, levels=15, origin='lower', alpha=0.7)

        if timestamps:
            if aarp_id:
                ax.set_title(f'{timestamps[frame]}_AARP_Id:{aarp_id}_channel_{channel}')
    ani = FuncAnimation(fig, update, frames = nframes, interval=50)
    ani.save(f'{filename}', writer='ffmpeg', fps=1)

def make_attribution_movie(filename, data:np.ndarray, channel, timestamps, vmax_frac=0.2, aarp_id=None):
    """
    Receives as input attribution data corresponding to single channel,
    associates timestamp values and aarp_id
    Generates a movie and saves it using the filename provided
    """
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
    """
    Receives a dictionary of ARs, the training metadata file and an AARP ID
    If the AR dict doesn't contain an entry with the given AARP id,
    creates a new Active Region object and loads it with data from the metadata file
    and adds it to the passed dictionary
    """
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
    images = np.where(images<0, np.zeros_like(images), images)
    # create a copy of the images before standardizing for visual plotting
    images_pre = images.copy()
    images = tf.image.per_image_standardization(images)
    attribution_masks  = get_attributions_mask(images, model, target_class_idx=label, args=args)
    images = np.expand_dims(images, axis=0)
    prediction = model.predict(images)

    return attribution_masks

def single_channel_attribution(model, images):
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
    parser.add_argument('--aarp-id', type=int, default=7304, help="AARP id for generating movie")
    parser.add_argument('--wavelength', type=int, help="Wavelength to use for generating animations")
    args = parser.parse_args()

    # force channels-first ordering
    backend.set_image_data_format('channels_first')

    with open(args.json_path) as json_file:
        data = json.load(json_file)

    channel_idx = next(int(k) for k,v in data['channels'].items() if v == args.wavelength)
    print("Channel index for given passband is", channel_idx)
    all_wavelengths = [94,
        131,
        171,
        193,
        211,
        304,
        335]

    ar_dict = {}

    aarps_ids_labels = [ (p['aarp_id'], p['label']) for p in data.get('test')]
    aarp_ids = [aarp_id for aarp_id, label in aarps_ids_labels]
    flared_aarp_ids = [ aarp_id for aarp_id, label in aarps_ids_labels if label == 1]
    print("Unique AARP_Ids in test", set(aarp_ids))
    print("Unique Flared AARP ids", set(flared_aarp_ids))
    add_observations_for_aarp(ar_dict, data, args.aarp_id )

    print(ar_dict.keys())
    first_key, first_value = next(iter(ar_dict.items()))
    print("First element:", first_key)

    # pick a particular channel
    ar_data_passband, timestamps = first_value.get_observation(args.wavelength)
    print("AARP Id", first_value.aarp_id)
    ar_data_passband = np.array(ar_data_passband)

    print("Label", first_value.label)
    # make movie from actual observations
    tempfile_name = next(tempfile._get_candidate_names())
    active_region.make_aia_movie(f'tmp{tempfile_name}_raw_movie_{args.aarp_id}_{args.wavelength}.mp4', ar_data_passband,
                                 wavelength=args.wavelength, timestamps = timestamps, aarp_id=first_value.aarp_id, label=first_value.label)

    model = get_compiled_model(args)
    model.load_weights(args.trained_model)

    alltimes = list(first_value._get_observation_generator(all_wavelengths))
    timestamps = [ timestamp for _, timestamp in alltimes]
    attribution_ts = []
    for multiband_obs,_ in alltimes:
        images = np.array(multiband_obs)
        #attr = single_attribution(model, images, first_value.label, args)
        #images, images_pre = preprocess_data(images)
        attribution_mask = get_attributions_mask(images, model, target_class_idx=0, args=args)
        print("Sum of intensities in attribution:", attribution_mask.numpy().sum())
        attribution_ts.append(attribution_mask)

    all_attribution_arr = np.array(attribution_ts)
    make_attribution_movie(f"tmp{tempfile_name}_attrb_movie_{args.aarp_id}.mp4", all_attribution_arr, channel=1, timestamps = timestamps, aarp_id=first_value.aarp_id)
    attribution_contour(f'tmp{tempfile_name}_raw_movie_{args.aarp_id}_{args.wavelength}.mp4', ar_data_passband, all_attribution_arr, wavelength=args.wavelength, channel=channel_idx, timestamps = timestamps, aarp_id=first_value.aarp_id)
