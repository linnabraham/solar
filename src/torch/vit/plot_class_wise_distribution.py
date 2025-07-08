import json
from aarp_ml.dataset import all_wavelengths
import torch
import matplotlib.pyplot as plt
import os
from vit.scripts.ig import single_aarp
from vit.scripts.predictions_analyze import dfs_from_metadata
from vit.scripts.class_wise_distribution import plot_intensity_distribution

if __name__=="__main__":
    with open('solar_dataset.json', 'r') as json_file:
        metadata = json.load(json_file)

    training_df, val_df, test_df = dfs_from_metadata(metadata)

    attributions_list_neg  = torch.load("data/intermediate-outs/attributions_neg.pt")
    attributions_list_pos  = torch.load("data/intermediate-outs/attributions_pos.pt")

    images_list_pos = []
    images_list_neg = []

    # Load images alone
    for aarp_id in test_df.query('label == 1').aarp_id.unique():
        print(aarp_id)
        aarp_id_df = test_df.query(f'aarp_id == {aarp_id}')
        s_aarp = single_aarp(aarp_id, aarp_id_df)
        s_images = s_aarp.get_images()
        images_list_pos.append(s_images)

    for aarp_id in test_df.query('label == 0').aarp_id.unique():
        print(aarp_id)
        aarp_id_df = test_df.query(f'aarp_id == {aarp_id}')
        s_aarp = single_aarp(aarp_id, aarp_id_df)
        s_images = s_aarp.get_images()
        images_list_neg.append(s_images)

    passband = 131
    percentile_levels = [50, 80, 90, 99]
    channel = all_wavelengths.index(passband)

    fig, ax = plot_intensity_distribution(images=(images_list_neg, images_list_pos),
                                attributions = (attributions_list_neg, attributions_list_pos),
                                percentile_levels=percentile_levels, passband=passband, x_range=(0,6),
                                nbins=30, alpha=0.4, figsize=(24,5), dpi=150)
    fig.savefig(f"plots/class_wise_int_dist_passband_{passband}.png", bbox_inches="tight")
    plt.close(fig)

    passband = 94
    percentile_levels = [50, 80, 90, 99, 99.9]
    x_range = (4,8)
    channel = all_wavelengths.index(passband)

    fig, ax = plot_intensity_distribution(images=(images_list_neg, images_list_pos),
                                attributions = (attributions_list_neg, attributions_list_pos),
                                percentile_levels=percentile_levels, passband=passband, x_range=x_range,
                                nbins=30, alpha=0.4, figsize=(24,5), dpi=150)

    fig.savefig(f"plots/class_wise_int_dist_passband_{passband}.png", bbox_inches="tight")
    plt.close(fig)
