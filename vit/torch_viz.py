#!/bin/env python
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
import os
from vit_pytorch import ViT
from torch_train import aia_euv, DeepFlare_ViT, CustomTransform
import pickle
from torchvision.transforms import v2
import json


def generate_plots(images, labels):
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

