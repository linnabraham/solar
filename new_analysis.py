import argparse
import os
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import tensorflow as tf
import sys
from joblib import Parallel, delayed
from active_region import active_region

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

#def parse_json(json_path):
#    with open(json_path) as f:
#        data = json.load(f)
#
#        x_test = [
#                [
#                [os.path.join(os.path.dirname(json_path), c[str(i)]) for i in range(7)]
#                , c['label'], c['aarp_id'], c['timestamp']
#                ]
#                 for c in data.get('test')
#                 ]
#        y_test = [p['label'] for p in data.get('test')]
#        #aarp_ids = [ p['aarp_id'] for p in data.get('test')]
#        #ts = [ p['timestamp'] for p in data.get('test')]
#
#
#        #return aarp_ids, ts
#        return x_test, y_test

def add_observations_for_aarp(active_regions_dict, data, aarp_id):
    relevant_entries = [entry for entry in data["test"] if entry["aarp_id"] == aarp_id]

    if aarp_id not in active_regions_dict:
        region = active_region(aarp_id, relevant_entries[0]["label"])  # Assuming all entries have the same label
        active_regions_dict[aarp_id] = region
    else:
        region = active_regions_dict[aarp_id]
    def process_entry(entry):
        timestamp = entry["timestamp"]
        for wavelength, fits_path in entry.items():
            if wavelength.isdigit():  # Check if the key is a digit (to exclude "label", "aarp_id", and "timestamp")
                region.add_observation(wavelength, timestamp, fits_path)
    # Add observations for each wavelength from the filtered entries
    num_jobs = min(len(relevant_entries), 10)
    Parallel(n_jobs=-1)(delayed(process_entry)(entry) for entry in relevant_entries)

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
    #fig = plot_attributions_v2(attribution_masks, images_pre)
    #plt.show()
    #plt.savefig("newsomething.png")

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-json_path', default="solar_dataset.json")
    parser.add_argument('-trained_model', default="outputs/best_model.h5")
    parser.add_argument('-input_shape', nargs='+', type=int, default=(512,512))
    args = parser.parse_args()

    #x_test, y_test = parse_json(args.json_path)

    with open(args.json_path) as json_file:
        data = json.load(json_file)

    ar_dict = {}
    st = time.time()

    aarps_ids_labels = [ (p['aarp_id'], p['label']) for p in data.get('test')]
    #print(aarps_ids_labels)
    aarp_ids = [aarp_id for aarp_id, label in aarps_ids_labels]
    print("Unique AARP_Ids in test", set(aarp_ids))
    add_observations_for_aarp(ar_dict, data, 57 )

    #for entry in data["test"]:
    #    aarp_id = entry["aarp_id"]
    #    add_observations_for_aarp(ar_dict, data, aarp_id)
    #    break

    #print(f"Took {time.time() - st} seconds")

    #count = 0
    #for entry in data.get("training"):
    #    #if count > 100:
    #    #    break
    #    aarp_id = entry["aarp_id"]
    #    if aarp_id != 377:
    #        continue
    #    #print("Processing AARP with id", aarp_id)
    #    label = entry["label"]
    #    if aarp_id not in ar_dict:
    #        region = active_region(aarp_id, label)
    #        ar_dict[aarp_id] = region
    #    else:
    #        region = ar_dict[aarp_id]

    #    for channel_num in range(7):
    #        timestamp = entry["timestamp"]
    #        channel = str(channel_num)
    #        fits_path = entry[channel]
    #        region.add_observation(channel,timestamp,fits_path)

    #    count += 1

    #print(len(ar_dict))
    #print(ar_dict.keys())
    first_key, first_value = next(iter(ar_dict.items()))
    print("First element:", first_key)
    ##print("Keys inside first element", first_value.data.keys())
    #print("Keys inside first timestamp", first_value.data['2011-07-30T15:42:02Z'].keys())
    #first_value_ts = list(first_value.data.keys())[0]
    #multiband_aarp = [ first_value.data[first_value_ts][i] for i in first_value.data[first_value_ts].keys()]
    #print(len(multiband_aarp))
    #print(multiband_aarp[0].shape)
    #print(multiband_aarp[4].shape)
    #multi_band_data = np.array(multiband_aarp)
    #print(multi_band_data.shape)



    #print("First element:", first_key, first_value.get_observation(94))

    #ar_data_94, timestamps = first_value.get_observation(94)
    ar_data_171, timestamps = first_value.get_observation(171)
    print("AARP Id", first_value.aarp_id)
    ar_data_171 = np.array(ar_data_171)
    #ar_data = np.array(ar_data)
    #print(ar_data.shape)
    #print(ar_data.min(), ar_data.max())
    #print(timestamps)

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

    #returnval = first_value._get_common_timestamps(all_wavelengths)
    alltimes = list(first_value._get_observation_generator(all_wavelengths))
    timestamps = [ timestamp for _, timestamp in alltimes]
    #print(timestamps)
    attribution_ts = []
    for multiband_obs,_ in alltimes:
        images = np.array(multiband_obs)
        attr = single_attribution(model, images, first_value.label, args)
        attribution_ts.append(attr)

    #print(len(all_attributions))
    all_attribution_arr = np.array(attribution_ts)
    make_attribution_movie("cool_movie.mp4", all_attribution_arr, channel=1, timestamps = timestamps, aarp_id=first_value.aarp_id)


    #returnval = next(first_value._get_observation_generator(all_wavelengths))
    #print("label", first_value.label)
    #images = np.array(returnval)
    #print(images.shape)
    #model = get_compiled_model(args)
    #model.load_weights(args.trained_model)
    ##print(model.summary())
    #attr = single_attribution(model, images, first_value.label, args)
    #print(attr.shape)

    #comb_data = combine_data("data/GOES_event_list.csv", "data/all_harps_with_noaa_ars.txt")
    #print(comb_data)
    #print(comb_data.dtypes)
    #print(comb_data.query('harpnum == 753'))




