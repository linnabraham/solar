import os
import json
from aarp_ml.dataset import all_wavelengths
import torch
import matplotlib.pyplot as plt
import scienceplots
plt.style.use(['no-latex'])
plt.rcParams.update({'font.size': 24})
import seaborn as sns
sns.set_theme()
from src.torch.vit.ig import single_aarp
from src.torch.vit.utils import dfs_from_metadata, get_metadata_from_json
from src.torch.vit.class_wise_distribution import plot_intensity_distribution

__all__ = [
    "load_images_for_label",
    "plot_distribution_for_passband",
    "plot_all_passbands",
]


def load_images_for_label(df, label: int):
    images_list = []
    for aarp_id in df.query(f'label == {label}').aarp_id.unique():
        print(aarp_id)
        aarp_id_df = df.query(f'aarp_id == {aarp_id}')
        s_aarp = single_aarp(aarp_id, aarp_id_df)
        s_images = s_aarp.get_images()
        images_list.append(s_images)
    return images_list

def plot_distribution_for_passband(
    passband: int,
    images_list_neg,
    images_list_pos,
    attributions_list_neg,
    attributions_list_pos,
    percentile_levels=None,
    x_range=None,
    nbins=30,
    alpha=0.4,
    figsize=(24, 5),
    dpi=150,
):
    """
    Generate and return intensity distribution plot for a single passband.
    Returns (fig, ax).
    """
    # Default configs per passband
    default_percentiles = {
        94:  [80, 90, 99, 99.9],
        131: [80, 90, 99, 99.9],
        171: [80, 90, 99, 99.9],
        193: [80, 90, 99, 99.9],
        211: [80, 90, 99, 99.9],
        304: [80, 90, 99, 99.9],
        335: [80, 90, 99, 99.9],
    }
    default_x_ranges = {
        94:  (0, 8),
        131: (0, 6),
        171: (4, 8),
        193: (4, 8),
        211: (3, 8),
        304: (3, 8),
        335: (0, 6),
    }

    if percentile_levels is None:
        percentile_levels = default_percentiles[passband]
    if x_range is None:
        x_range = default_x_ranges[passband]

    print(f"Generating distribution plot for passband={passband}, "
          f"percentiles={percentile_levels}, x_range={x_range}")

    fig, ax = plot_intensity_distribution(
        images=(images_list_neg, images_list_pos),
        attributions=(attributions_list_neg, attributions_list_pos),
        percentile_levels=percentile_levels,
        passband=passband,
        x_range=x_range,
        nbins=nbins,
        alpha=alpha,
        figsize=figsize,
        dpi=dpi,
    )
    return fig, ax

def plot_all_passbands(df, images, attributions, passbands=None, save_dir=None):
    images_list_neg, images_list_pos = images
    attributions_list_neg, attributions_list_pos = attributions
    if passbands is None:
        passbands = df.columns  # or your `all_wavelengths`
    plots = {}
    for pb in passbands:
        fig, ax = plot_distribution_for_passband(
        pb,
        images_list_neg, images_list_pos,
        attributions_list_neg, attributions_list_pos
        )
        plots[pb] = (fig, ax)

        # save immediately if requested
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            #outpath = os.path.join(save_dir, f"{pb}.png")
            outpath = os.path.join(save_dir, f"{pb}.pdf")
            fig.savefig(outpath, format="pdf", dpi=300, bbox_inches="tight")
            plt.close(fig)  # optional: free memory right away
    return plots


if __name__=="__main__":
    metadata = get_metadata_from_json('solar_dataset.json')

    training_df, val_df, test_df = dfs_from_metadata(metadata)

    attributions_list_neg  = torch.load("data/intermediate-outs/attributions_neg.pt")
    attributions_list_pos  = torch.load("data/intermediate-outs/attributions_pos.pt")

    # Load images alone
    images_list_pos = load_images_for_label(test_df, label=1)
    images_list_neg = load_images_for_label(test_df, label=0)

    images = (images_list_neg, images_list_pos)
    attributions = (attributions_list_neg, attributions_list_pos)

    plots = plot_all_passbands(test_df, images, attributions, passbands=all_wavelengths, save_dir="plots/distribution")
