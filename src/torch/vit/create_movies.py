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
from src.torch.vit.plot_contour_image import run_pred_and_ig

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

if __name__=="__main__":
    print(animation.writers.list())
    config = TrainingConfig(json_path="solar_dataset.json", stats_file="stats.pkl")
    config.trained_model_path = "output/glad-shape-197/trained_model.pth"
    print(config)
    metadata, model, transform, device = get_data_model(config)
    training_df, val_df, test_df = dfs_from_metadata(metadata)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    aarp_id = 3563
    s_images, attributions = run_pred_and_ig(aarp_id, val_df, transform, model, device)
    attributions_arr = np.array(attributions)
    channel = 1
    passband_attributions = attributions_arr[:, channel, :,:]
    passband = 131
    filename = f"tests_outputs/movie_{passband}.mp4"
    make_attribution_movie(
        images=s_images,
        attributions=passband_attributions,
        passband=passband,
        filename=filename,
        channel=channel,
        fps=1
    )
