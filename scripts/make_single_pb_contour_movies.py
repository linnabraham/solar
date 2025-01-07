import os
import sys
import argparse
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(base_path)
import pickle
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from aarp_ml.dataset import aarp_dataset
from aarp_ml.model import training
from aarp_ml.integrated_gradients.aarp_ig import get_attribution_sequence

def plot_aia_image(data, passband, **kwargs):
    """Plots an AIA image on the given axis."""
    aia_cmap = plt.get_cmap(f'sdoaia{passband}')
    if 'vmax_percentile' in kwargs:
        kwargs['vmax'] = np.percentile(data, kwargs['vmax_percentile'])
        kwargs.pop('vmax_percentile')
    im = ax.imshow(data, cmap=aia_cmap, origin='lower', **kwargs)
    return im

def plot_contours(ax, data, **kwargs):
    """Plots contours on the given axis."""
    contour = ax.contour(data, linewidths=0.5, **kwargs)
    for i, c in enumerate(contour.collections):
        if i < 1:
            c.remove()
    return contour

def init():
    """Sets up the initial plot elements."""
    ax.set_title("AIA Image with Contours")
    ax.axis("off")
    # Plot the first frame
    im = plot_aia_image(seq_pb[0], passband, vmax_percentile=99.5)
    contours = plot_contours(ax, attbn_pb[0])
    return [im] + contours.collections

def update(frame_idx):
    """Updates the image and contours for each frame."""
    ax.clear()
    ax.axis("off")
    im = plot_aia_image(seq_pb[frame_idx], passband, vmax_percentile=99.5)
    contours = plot_contours(ax, attbn_pb[frame_idx])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--passband", type=int, required=True)
    parser.add_argument("--aarp-id", type=int, required=True)
    parser.add_argument("--trained-model", required=True)
    parser.add_argument("--vmax-percentile", type=float, default=99.5)
    parser.add_argument("--subset", default="test")
    args = parser.parse_args()

    stats_file=os.path.join(base_path,"stats.pkl")
    json_file = os.path.join(base_path, "solar_dataset.json")
    ds = aarp_dataset(json_file)
    subset = ds.get_subset(args.subset)
    print(f"Using {stats_file} and {json_file}")
    print(f"Using {args.vmax_percentile} as vmax percentile value")
    fig, ax = plt.subplots()
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
    model_id = args.trained_model.split("/")[-2]
    movie_save_path = f"contour_animation_{args.aarp_id}_{args.passband}_{model_id}.mp4"
    print(f"Creating movie and saving to {movie_save_path}")
    passbands = subset.all_wavelengths
    passband = args.passband
    channel = passbands.index(args.passband)

    aarp_seq = subset.create_aarp_sequence(args.aarp_id)

    model = training(ds, stats_file).get_trained_model(args.trained_model).model
    att_seq = get_attribution_sequence(aarp_seq, model)
    attbn_pb = att_seq.images[:,channel,:,:]
    seq_pb = aarp_seq.images[:,channel,:,:]

    ani = FuncAnimation(fig, update, frames=len(seq_pb),blit=False)
    ani.save(movie_save_path, fps=10, writer='ffmpeg')
