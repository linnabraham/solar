import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
import torch
import gc
from itertools import islice
from torch.utils.data import DataLoader, TensorDataset
from src.torch.vit.train import TrainingConfig
from src.torch.vit.utils import get_data_model, dfs_from_metadata
from src.torch.vit.ig import single_aarp, do_ig
from src.torch.vit.class_wise_distribution import run_pred_and_ig

def make_attribution_movie(
    images,               # numpy array, shape (T, H, W) or (T, C, H, W)
    attributions,         # numpy array, shape (T, H, W)
    passband,             # e.g., 131
    filename,             # output filename, e.g., 'output.mp4'
    channel=0,            # if images are 4D, selects which channel to use
    vmax_percentile=99.9, # percentile for image scaling
    contour_levels=5,     # number of contour levels (draws only topmost)
    fps=5,                # frames per second in output
    interval=50           # interval between frames in milliseconds
):
    # Select channel if 4D
    if images.ndim == 4:
        images = images[:, channel, :, :]
    if attributions.ndim == 4:
        attributions = attributions[:, channel, :, :]

    nframes = images.shape[0]

    fig, ax = plt.subplots()
    vmax = np.percentile(images, vmax_percentile)
    im = ax.imshow(images[0], origin='lower', vmax=vmax)

    # Compute contour levels globally
    min_val = np.min(attributions)
    max_val = np.max(attributions)
    levels = np.linspace(min_val, max_val, num=contour_levels+2)[1:-1]  # exclude min/max
    contour_obj = [None]  # holds the contour so we can remove it later

    def update(frame):
        im.set_array(images[frame])
        if contour_obj[0] is not None:
            for coll in contour_obj[0].collections:
                coll.remove()

        saliency = attributions[frame]
        contour_obj[0] = ax.contour(saliency, levels=levels[-1:], colors='red', linewidths=1.5)
    ani = FuncAnimation(fig, update, frames=nframes, interval=interval)
    ani.save(filename, writer='ffmpeg', fps=fps)
    plt.close(fig)

if __name__=="__main__":
    import argparse
    from pathlib import Path
    from aarp_ml.dataset import all_wavelengths

    parser = argparse.ArgumentParser()
    parser.add_argument("--json-path",   default="solar_dataset.json")
    parser.add_argument("--output-dir",  default="plots/attribution_movies")
    parser.add_argument("--passband",    type=int, default=131)
    args = parser.parse_args()

    config = TrainingConfig(json_path=args.json_path, stats_file="stats.pkl")
    config.trained_model_path = "outputs/glad-shape-197/trained_model.pth"
    metadata, model, transform, device = get_data_model(config)
    training_df, val_df, test_df = dfs_from_metadata(metadata)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    channel = all_wavelengths.index(args.passband)

    for df, split in [(training_df, "training"), (val_df, "validation"), (test_df, "test")]:
        if df.empty:
            continue
        for aarp_id in df.aarp_id.unique():
            output_path = os.path.join(args.output_dir, f"{aarp_id}.mp4")
            aarp_id_df = df.query(f"aarp_id == {aarp_id}")
            s_images, attributions = run_pred_and_ig(aarp_id, aarp_id_df, transform, model, device)
            attributions_arr = np.array(attributions)
            passband_attributions = attributions_arr[:, channel, :, :]
            make_attribution_movie(
                images=s_images,
                attributions=passband_attributions,
                passband=args.passband,
                filename=output_path,
                channel=channel,
                fps=1
            )
            del s_images, attributions, attributions_arr, passband_attributions
            gc.collect()
            torch.cuda.empty_cache()
