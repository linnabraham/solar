#!/bin/env python
"""
Script for computing the mean and std of the training dataset once to be used for z-score normalization
in the training script.
Be careful of running the script again since now the CustomTransform function might already have the z-score
normalization included.
Two functions were written but the second one seemed slightly faster. Results were almost the same in both methods.
"""
import argparse
import torch
from torch.utils.data import DataLoader
from torch_train import aia_euv, CustomTransform
from torchvision import transforms
from tqdm import tqdm
import time
import pickle

def compute_stats_a(num_channels, train_loader):
    channel_sum = torch.zeros(num_channels, device=device)
    channel_sum_of_squares = torch.zeros(num_channels, device=device)
    channel_count = 0
    for batch,_ in tqdm(train_loader, total=len(train_loader), leave=False):
        batch = batch.to(device)
        batch_size = batch.size(0)
        channel_sum_of_squares += (batch ** 2).sum(dim=(0, 2, 3))
        channel_sum += batch.sum(dim=(0, 2, 3))
        channel_count += batch_size * batch.size(2) * batch.size(3)
    channel_variance = (channel_sum_of_squares / channel_count) - ((channel_sum / channel_count) ** 2)
    channel_std = torch.sqrt(channel_variance)
    channel_mean = channel_sum/channel_count
    return channel_mean, channel_std

def compute_stats_b(num_channels, train_loader):
    data_mean = torch.zeros(num_channels, device=device)
    data_std = torch.zeros(num_channels, device=device)
    channel_count = 0
    i = 0
    for batch,_ in tqdm(train_loader, total=len(train_loader), leave=False):
        batch = batch.to(device)
        batch_size = batch.size(0)
        data_mean += batch.mean(dim=(0,2,3))
        data_std += batch.std(dim=(0,2,3))
        channel_count += batch_size
        i+=1
    return data_mean/i, data_std/i

if __name__== "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-batch-size", "--batch-size", type=int, default=128)
    args = parser.parse_args()

    train_dataset = aia_euv('../solar_dataset.json', subset='training', transform=transforms.Compose([CustomTransform()]))
    train_loader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    num_channels = 7
    t1 = time.time()

    data_mean, data_std = compute_stats_b(num_channels, train_loader)

    print("Time taken", time.time() - t1)

    stats = {
            'mean' : {f'channel_{i}': data_mean.cpu().numpy()[i] for i in range(num_channels)},
            'std' : {f'channel_{i}': data_std.cpu().numpy()[i] for i in range(num_channels)}
            }

    with open('stats.pkl', 'wb') as f:
        pickle.dump(stats, f)
