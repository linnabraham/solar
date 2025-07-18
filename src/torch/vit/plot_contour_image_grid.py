import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import math
import torch
from src.torch.vit.ig import single_aarp
from src.torch.vit.train import TrainingConfig
from src.torch.vit.utils import (get_data_model, dfs_from_metadata,
                                get_attribution_for_image)

def plot_aia_image_grid(
    images,
    passbands,
    cols=4,
    gap=0,
    dpi=100,
    vmax_percentile=None,
    show=False,
    saliency=None,
    contour_config=None
):
    """
    Plot a grid of AIA images, each with its corresponding colormap and optional saliency contours.

    Parameters
    ----------
    images : np.ndarray
        Array of shape [n, H, W], one 2D image per passband.
    passbands : list of int or str
        List of AIA wavelengths matching the order of `images`.
    cols : int
        Number of columns in the image grid.
    gap : int
        Pixel gap between images.
    dpi : int
        Resolution of the output figure.
    vmax_percentile : float or None
        If given, compute vmax for each image from this percentile.
    show : bool
        If True, displays the figure; otherwise returns it.
    saliency : np.ndarray or None
        Optional array of shape [n, H, W] for contour overlays.
    contour_config : dict or None
        Dictionary mapping passbands to contour parameters. Each value should be a dict like:
        {
            "num_levels": 85,
            "color": "blue",
            "line_width": 1.5,
            "use_last_n": 5
        }

    Returns
    -------
    fig : matplotlib.figure.Figure or None
        The resulting figure if show=False; otherwise None.
    """
    assert images.shape[0] == len(passbands), "Number of images and passbands must match"
    n, H, W = images.shape
    rows = math.ceil(n / cols)

    total_width_px = cols * W + (cols - 1) * gap
    total_height_px = rows * H + (rows - 1) * gap
    figsize = (total_width_px / dpi, total_height_px / dpi)

    fig = plt.figure(figsize=figsize, dpi=dpi)

    for idx, (image, passband) in enumerate(zip(images, passbands)):
        row = idx // cols
        col = idx % cols

        left = (col * (W + gap)) / total_width_px
        bottom = 1 - ((row + 1) * H + row * gap) / total_height_px
        width = W / total_width_px
        height = H / total_height_px

        ax = fig.add_axes([left, bottom, width, height])

        # Plot the base image
        aia_cmap = matplotlib.colormaps[f'sdoaia{passband}']
        if vmax_percentile is not None:
            vmax = np.percentile(image, vmax_percentile)
        else:
            vmax = None
        im = ax.imshow(image, cmap=aia_cmap, origin='lower', vmax=vmax)
        ax.axis("off")

        # Optionally plot contours if saliency is given
        if saliency is not None:
            s = saliency[idx]
            if contour_config and passband in contour_config:
                cfg = contour_config[passband]
                num_levels = cfg.get("num_levels", 10)
                color = cfg.get("color", "red")
                line_width = cfg.get("line_width", 1.0)
                use_last_n = cfg.get("use_last_n", 3)

                levels = np.linspace(np.min(s), np.max(s), num=num_levels)
                ax.contour(
                    s,
                    levels=levels[-use_last_n:],
                    colors=color,
                    linewidths=line_width,
                    origin='lower'
                )

    if show:
        plt.show()
    else:
        plt.close(fig)
        return fig

if __name__=="__main__":
    config = TrainingConfig(json_path="solar_dataset.json", stats_file="stats.pkl")
    config.trained_model_path = "outputs/glad-shape-197/trained_model.pth"
    metadata, model, transform, device = get_data_model(config)
    training_df, val_df, test_df = dfs_from_metadata(metadata)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    aarp_id = 3563
    channel = 1
    t_idx = 12

    aarp_id_df = val_df.query(f"aarp_id=={aarp_id}")
    s_aarp = single_aarp(aarp_id, aarp_id_df)
    s_images = s_aarp.get_images()
    image = s_images[t_idx]
    ig_out = get_attribution_for_image(image, s_aarp.label, transform, device, model, ib_size=1)
    saliency = ig_out[channel]

    fig = plot_aia_image_grid(
        images=s_images[t_idx],           # shape (n_channels, H, W)
        passbands=[94, 131, 171, 193, 211, 304, 335],
        saliency=ig_out,           # shape (n_channels, H, W)
        contour_config={
            94: {"num_levels": 35, "color": "red", "line_width": 1.5, "use_last_n": 2},
            131: {"num_levels": 10, "color": "red", "line_width": 1.5, "use_last_n": 2},
            171: {"num_levels": 10, "color": "red", "line_width": 1.0, "use_last_n": 3},
            193: {"num_levels": 20, "color": "blue", "line_width": 1.5, "use_last_n": 2},
            211: {"num_levels": 20, "color": "blue", "line_width": 1.5, "use_last_n": 3},
            304: {"num_levels": 30, "color": "blue", "line_width": 1.5, "use_last_n": 2},
            335: {"num_levels": 25, "color": "red", "line_width": 1.5, "use_last_n": 2},
        },
        vmax_percentile=99.9,
        show=False
    )
    fig.savefig("plots/attribution_contour_grid.png", bbox_inches="tight", dpi=300)
