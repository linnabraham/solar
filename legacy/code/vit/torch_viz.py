#!/bin/env python
import os
import sys
cwd = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(cwd, ".."))
sys.path.append(parent_dir)
import argparse
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler
from captum.attr import IntegratedGradients
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import sunpy.visualization.colormaps as cm
import random
import uuid
from vit_pytorch import ViT
from torch_train import aia_euv, DeepFlare_ViT, CustomTransform
import pickle
from torchvision.transforms import v2
import json
import tempfile
from matplotlib.animation import FuncAnimation
import time
from active_region import add_observations_for_aarp
from utils.aia_metadata import all_wavelengths

def single_attribution(image, label, channel):
    """
    Generate attribution for single image and for single channel
    """
    #image_arr = image.squeeze().cpu().detach().numpy()
    #image_channel = image_arr[channel]
    image = image.clone().to(device)
    label = label.clone().to(device)
    baseline_zero = torch.zeros_like(image)
    ig = IntegratedGradients(model)
    ig_b0, _ = ig.attribute(image, baseline_zero, target=label, n_steps=100,
                                        internal_batch_size=7, return_convergence_delta=True)
    ig_b0 = ig_b0.squeeze().detach().cpu().numpy()

    ig_b0 = ig_b0[channel,:,:]
    return ig_b0

# TODO: this function is copied from the tensorflow version and modified keeping mind ordering of channels
def make_attribution_movie(filename, attributions:np.ndarray, obs_data:np.ndarray, timestamps, vmax_frac=0.2, aarp_id=None):
    nframes = attributions.shape[0]
    mask_max = np.max(attributions)
    fig, ax = plt.subplots()

    cmap_key = 'sdoaia'+str(args.wavelength)
    sdoaia_cmap = matplotlib.colormaps[cmap_key]
    im1 = ax.imshow(obs_data[0,:,:], cmap=sdoaia_cmap, origin='lower')
    im2 = ax.imshow(attributions[0,:,:], cmap=plt.cm.jet, origin='lower', alpha=0.4)
    cbar = fig.colorbar(im2, ax=ax)
    threshold = np.percentile(attributions, 90)
    print("90 percentile value of attribution is ", threshold)
    def update(frame):
        im1.set_array(obs_data[frame,:,:])
        im_masked = np.ma.masked_where(attributions[frame,:,:] < threshold, attributions[frame,:,:]) 
        im2.set_array(attributions[frame,:,:])
        cbar.update_normal(im2)
        if timestamps:
            if aarp_id:
                ax.set_title(f'{timestamps[frame]}_AARP_Id:{aarp_id}_passband_{args.wavelength}')
    ani = FuncAnimation(fig, update, frames = nframes, interval=50)
    ani.save(f'{filename}', writer='ffmpeg', fps=5)

def generate_plots(images, labels, channel):
    """
    Generate plots using IG attributions
    Input: image and label batch of size 1
    Output: The IG attribution for the 0th channel AIA 94 overlayed on the image with transparency
    """
    aia_cmaps = {0:'sdoaia94',
                    1:'sdoaia131',
                    2:'sdoaia171',
                    3:'sdoaia193',
                    4:'sdoaia211',
                    5:'sdoaia304',
                    6:'sdoaia335'}

    with open(args.json_path) as json_file:
        data = json.load(json_file)
    channel = next(int(k) for k,v in data['channels'].items() if v == args.wavelength)

    cmap = matplotlib.colormaps[aia_cmaps[channel]]

    images_arr = images.squeeze().cpu().detach().numpy()
    image = images_arr[channel]

    plt.imshow(np.sqrt(np.where(image<0, 0, image)), cmap=cmap, origin='lower')

    images = images.clone().to(device)
    labels = labels.clone().to(device)

    # Generate baselines
    baseline_zero = torch.zeros_like(images)
    #baseline_one = torch.ones_like(images)

    # Compute attributions using Integrated Gradients
    ig = IntegratedGradients(model)
    ig_b0, _ = ig.attribute(images, baseline_zero, target=labels, n_steps=100,
                                        internal_batch_size=1, return_convergence_delta=True)
    ig_b0 = ig_b0.squeeze().detach().cpu().numpy()

    ig_b0 = ig_b0[channel,:,:]

    #vmin= np.percentile(ig_b0,98)
    #vmax = np.percentile(ig_b0, 99)

    alpha = 0.4

    #plt.imshow(ig_b0, vmin=vmin, vmax=vmax, alpha=alpha, cmap=plt.cm.jet, origin='lower')
    plt.imshow(ig_b0, alpha=alpha, cmap=plt.cm.jet, origin='lower')
    plt.colorbar()

def plot_random(val_loader, args):
def plot_single_aarp(active_region, channel):
    """
    Generate movie combining all attributions from a single aarp id ordered in time for a single channel
    """
    obs_alltimes = list(active_region._get_observation_generator(all_wavelengths))
    timestamps = [timestamp for _, timestamp in obs_alltimes]
    attribution_masks = []
    start = time.time()
    for single_obs, _ in obs_alltimes:
        image = torch.tensor(np.array(single_obs),  dtype=torch.float32)
        image = image.unsqueeze(0)
        label = torch.tensor(active_region.label)
        label = label.unsqueeze(0)
        single_channel_attrib = single_attribution(image, label, channel)
        attribution_masks.append(single_channel_attrib)
    end = time.time()
    print("Number of frames", len(attribution_masks), "time taken:", np.round(end-start,4))
    attribution_masks_arr = np.array(attribution_masks)
    tempfile_name = next(tempfile._get_candidate_names())
    obs_data_list = [ single_obs for single_obs, _ in obs_alltimes ]
    obs_data = np.array(obs_data_list)[:,channel,:,:]
    make_attribution_movie(f"tmp{tempfile_name}_attrb_movie_{args.aarp_id}.mp4", attribution_masks_arr, obs_data,
                           timestamps = timestamps, aarp_id=active_region.aarp_id)

    count = 0
    for images, labels in val_loader:
        assert args.batch_size == 1
        images_arr = images.cpu().detach().numpy()
        labels_arr = labels.cpu().detach().numpy()
        label = int(labels_arr[0])

        image_max = np.max(images_arr)
        image_min = np.min(images_arr)

        if label == 1:
            generate_plots(images, labels)
            plt.savefig(os.path.join(tmp_output,f"ig_b0_v3_{count}.png"))
            plt.close()
        else:
            # skip non-flared ARs for now
            continue
        #plt.savefig(f"ig_b0_v3_{label}_{count}.png")
        #plt.close()
        count += 1

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-json-path', '--json-path')
    parser.add_argument('-batch-size', '--batch-size', type=int, default=1)
    parser.add_argument('-saved-model', '--saved-model')
    parser.add_argument('--wavelength', type=int, help="Wavelength to use for generating animations")
    parser.add_argument('--aarp-id', type=int)
    args = parser.parse_args()

    torch.manual_seed(42)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.reset_peak_memory_stats()

    # read the mean and std computed over the whole data and pickled to disk
    with open('stats.pkl', 'rb') as f:
        stats = pickle.load(f)

    means = [stats['mean'][f'channel_{i}'] for i in range(7)]
    stds = [stats['std'][f'channel_{i}'] for i in range(7)]

    #val_ds = aia_euv(args.json_path, subset='validation')

    # use the transforms used during training here as well
    val_ds = aia_euv(args.json_path, subset='validation', transform=v2.Compose([CustomTransform(means, stds)]))
    val_loader = DataLoader(val_ds, batch_size = args.batch_size, shuffle=False)

    num_samples = 50  # Change this to the desired number of samples

    # Create a new data loader with the RandomSampler
    sampler = RandomSampler(val_ds, num_samples=num_samples)
    sampled_validation_loader = DataLoader(val_ds, batch_size=args.batch_size, sampler=sampler)
    print(f"Randomly sampling {len(sampled_validation_loader)} items from dataset")

    model = DeepFlare_ViT(height=512, n_classes=2, n_passbands=7).model
    model.load_state_dict(torch.load(args.saved_model))
    model.to(device)
    model.eval()


    tmp_output = str(uuid.uuid4())[:8]
    os.makedirs(tmp_output)
    print(f"Creating directory {tmp_output} for outputs")
    plot_random(sampled_validation_loader, args)
    with open(args.json_path) as json_file:
        data = json.load(json_file)
    channel = next(int(k) for k,v in data['channels'].items() if v == args.wavelength)

    ar_dict = {}
    add_observations_for_aarp(ar_dict, data, args.aarp_id)
    first_key, first_value = next(iter(ar_dict.items()))
    plot_single_aarp(first_value, channel)
