#!/bin/env python

import numpy as np
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
from skimage import measure
import argparse
from glob import glob
import sys
from concurrent.futures import ThreadPoolExecutor

def investigate_contours(attributions):
    channel = 2
    channel_attribution = attributions[:,:,:,channel]
    threshold = np.percentile(channel_attribution, 90)
    for frame in range(channel_attribution.shape[0]):
        attribution_im = channel_attribution[frame,:,:]
        #print(attribution_im.shape)
        print(np.sum(attribution_im))
        im_masked = np.ma.masked_where(attribution_im < threshold, attribution_im)
        print(np.sum(im_masked))
        print(np.unique(im_masked))
        print(len(np.unique(im_masked)))
        contour = plt.contour(im_masked)
        print(contour.levels)
        #labeled_array, num_features = ndimage.label(im_masked)
        #print(labeled_array.shape, num_features)
        plt.imshow(im_masked)
        #for level in contour.levels:
        #    contours = measure.find_contours(im_masked, level=level)
        #    print("Len of contours", len(contours))
        #    for contour in contours:
        #        plt.plot(contour[:, 1], contour[:, 0], 'r')  # plot contours in red
        contours = measure.find_contours(im_masked, level=contour.levels[-10])
        ncontour = 0
        for contour in contours:
            plt.plot(contour[:, 1], contour[:, 0], 'r')  # plot contours in red

            plt.savefig(f"attribution_contour_y_{ncontour}.png")
            ncontour += 1
        break

def make_distplots(attributions, channel, aarp_id):
    channel_attribution = attributions[:,:,:,channel]
    frame = channel_attribution[0,:,:]
    print(np.min(frame), np.max(frame))
    values = frame.flatten()
    values = values[values > 0]
    values = np.log(values)
    print(np.percentile(values, 99.8))
    print(np.percentile(values, 99))
    print(np.percentile(values, 95))
    print(np.percentile(values, 90))
    print(np.percentile(values, 10))
    #plt.hist(values, bins=20 )
    plt.hist(values)
    #plt.hist(values, bins=20, range=(np.percentile(values,90),np.percentile(values,100)))
    plt.title("Distribution of attribution(log transformed) for single AARP and passband")
    plt.savefig(f"frame_dist_aarp_{aarp_id}.png")

def plot_intensity_dist():
    def load_and_process(file_path):
        intensities = np.load(file_path)
        return intensities.flatten()

    npyfiles_1 = glob(f"{args.dir_path}/1/*")

    # Parallel file loading
    with ThreadPoolExecutor() as executor:
            results = list(executor.map(load_and_process, npyfiles_1))

            values = np.concatenate(results)

    print(values.shape)

    values = values[values > 0]
    values = np.log(values)
    plt.hist(values, bins=20, label='flaring', density=True, histtype='stepfilled')

    npyfiles_0 = glob(f"{args.dir_path}/0/*")

    # Parallel file loading
    with ThreadPoolExecutor() as executor:
            results = list(executor.map(load_and_process, npyfiles_0))

            # Concatenate all arrays together once loaded
            values_0 = np.concatenate(results)

    print(values_0.shape)

    values_0 = values_0[values_0 > 0]
    values_0 = np.log(values_0)
    plt.hist(values_0, bins=20, alpha=0.6, histtype='stepfilled', density=True, label='non-flaring')
    plt.legend()
    plt.title("Distribution of AIA intensities of AARPS")
    plt.savefig(f"dist_aarp_intensities_combined.png")

def plot_intensity_with_attribution(percentile_level):

    def load_and_process(file_path):
        data = np.load(file_path)
        return data

    def process_label(label):

        att = glob(f"{args.att_dir_path}/{label}/*")
        with ThreadPoolExecutor() as executor:
                results = list(executor.map(load_and_process, att))
                att_values = np.concatenate(results)

        intensities = glob(f"{args.int_dir_path}/{label}/*")
        with ThreadPoolExecutor() as executor:
                results = list(executor.map(load_and_process, intensities))
                int_values = np.concatenate(results)

        att_values_ch_2 = att_values[:,:,:,2]
        int_values_ch_2 = int_values[:,2, :,:]

        # import pdb; pdb.set_trace()
        # do log transform
        # Set a constant smaller than the smallest positive value
        min_pos_value = np.min(att_values_ch_2[att_values_ch_2 > 0])
        epsilon = min_pos_value * 1e-5
        safe_attributions = np.where(att_values_ch_2 > 0, att_values_ch_2, epsilon)
        log_attributions = np.log(safe_attributions)

        threshold = np.percentile(log_attributions, percentile_level)
        print(threshold)

        # return intensities values corresponding to those pixels in attribution map 
        # greater than the threshold and 0 elsewhere
        int_selected = np.where(log_attributions > threshold, int_values_ch_2, 0)

        min_pos_value = np.min(int_selected[int_selected > 0])
        epsilon = min_pos_value * 1e-5
        safe_images = np.where(int_selected > 0, int_selected, epsilon)
        log_intensities = np.log(safe_images)

        print("Sum of log intensities", log_intensities.sum())
        return log_intensities.flatten()

    values = process_label(label=1)
    plt.hist(values, bins=20, histtype='stepfilled', density=True, label='flaring')
    values_0 = process_label(label=0)
    plt.hist(values_0, bins=20, histtype='stepfilled', density=True, alpha=0.6, label='non-flaring')
    plt.title(f"Distribution of intensities corresponding to attribution at {percentile_level}_percentile")
    plt.legend()
    plt.savefig("dist_int_with_att.png")

def plot_attribution_dist():

    def load_and_process(file_path):
        attrib_data = np.load(file_path)
        return attrib_data.flatten()

    npyfiles_1 = glob(f"{args.dir_path}/1/*")

    # Parallel file loading
    with ThreadPoolExecutor() as executor:
            results = list(executor.map(load_and_process, npyfiles_1))

            values = np.concatenate(results)

    print(values.shape)

    values = values[values > 0]
    values = np.log(values)
    threshold = np.percentile(values, 20)
    values = values[values>threshold]
    plt.hist(values, bins=20, histtype='stepfilled', density=True, label='flaring')

    npyfiles_0 = glob(f"{args.dir_path}/0/*")

    # Parallel file loading
    with ThreadPoolExecutor() as executor:
            results = list(executor.map(load_and_process, npyfiles_0))

            # Concatenate all arrays together once loaded
            values_0 = np.concatenate(results)

    print(values_0.shape)

    values_0 = values_0[values_0 > 0]
    values_0 = np.log(values_0)
    threshold = np.percentile(values, 20)
    values = values[values>threshold]
    plt.hist(values_0, bins=20, histtype='stepfilled', density=True, alpha=0.6, label='non-flaring')
    plt.title("Distribution of ML attributions")
    plt.legend()
    plt.savefig(f"dist_aarp_combined.png")


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file-path', help="Path to file containing attribution for a single AARP")
    parser.add_argument('--att-dir-path', help="Path to directory containing attributions for all AARPs")
    parser.add_argument('--int-dir-path', help="Path to directory containing intensities for all AARPs")
    args = parser.parse_args()

    if args.file_path is not None:
        p1 = args.file_path.split('aarp_')[1]
        aarp_id = int(p1.split('.npy')[0])
        attributions = np.load(args.file_path)
        make_distplots(attributions, aarp_id, channel=2)
    # plot_intensity_dist()
    plot_attribution_dist()
    # percentile_level = 90
    # print("Using percentile level", percentile_level)
    # plot_intensity_with_attribution(percentile_level)
