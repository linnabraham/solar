import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import math
import torch
from typing import Optional, Dict, List, Tuple
from src.torch.vit.ig import single_aarp
from src.torch.vit.train import TrainingConfig
from src.torch.vit.utils import (get_data_model, dfs_from_metadata,
                                get_attribution_for_image, needs_resize)

VALID_MODEL_TYPES = {"vit": "deepflare_vit", "vit-pretrained": "vit_pretrained"}

# ==================== Module Constants ====================
# Default contour configuration for each AIA passband (wavelength)
CONTOUR_CONFIG = {
    94: {"num_levels": 35, "color": "red", "line_width": 1.5, "use_last_n": 2},
    131: {"num_levels": 10, "color": "red", "line_width": 1.5, "use_last_n": 2},
    171: {"num_levels": 10, "color": "red", "line_width": 1.0, "use_last_n": 3},
    193: {"num_levels": 20, "color": "blue", "line_width": 1.5, "use_last_n": 2},
    211: {"num_levels": 20, "color": "blue", "line_width": 1.5, "use_last_n": 3},
    304: {"num_levels": 30, "color": "blue", "line_width": 1.5, "use_last_n": 2},
    335: {"num_levels": 25, "color": "red", "line_width": 1.5, "use_last_n": 2},
}

DEFAULT_COLS = 4
DEFAULT_GAP = 0
DEFAULT_DPI = 100
DEFAULT_VMAX_PERCENTILE = 99.9

def plot_aia_image_grid(
    images: np.ndarray,
    passbands: List[int],
    cols: int = DEFAULT_COLS,
    gap: int = DEFAULT_GAP,
    dpi: int = DEFAULT_DPI,
    vmax_percentile: Optional[float] = None,
    show: bool = False,
    saliency: Optional[np.ndarray] = None,
    contour_config: Optional[Dict] = None
) -> Optional[plt.Figure]:
    """Plot a grid of AIA images with optional saliency contours.
    
    Creates a grid visualization of AIA (Solar Dynamics Observatory) images,
    each with its corresponding SDO/AIA colormap. Optionally overlays contour
    lines from attribution/saliency maps to highlight important regions.
    
    Args:
        images (np.ndarray): Array of shape [n, H, W] containing one 2D image
            per passband. H and W are image height/width in pixels.
        passbands (List[int]): List of AIA wavelengths (e.g., [94, 131, 171, ...])
            matching the order of images. Used to select correct SDO/AIA colormap.
        cols (int, optional): Number of columns in the grid. Defaults to 4.
        gap (int, optional): Pixel gap between images in the grid. Defaults to 0.
        dpi (int, optional): Resolution in dots per inch. Defaults to 100.
        vmax_percentile (float, optional): If provided, normalize image colors to
            this percentile value (e.g., 99.9 scales colors to 99.9th percentile).
            If None, uses automatic scaling. Defaults to None.
        show (bool, optional): If True, displays the figure; if False, returns it.
            Defaults to False.
        saliency (np.ndarray, optional): Attribution/saliency array of shape [n, H, W]
            for overlaying attribution contours. If None, no contours are drawn.
            Defaults to None.
        contour_config (Dict, optional): Dictionary mapping passbands to contour 
            parameters. Example structure:
            {
                94: {"num_levels": 35, "color": "red", "line_width": 1.5, "use_last_n": 2},
                131: {"num_levels": 10, "color": "red", ...},
                ...
            }
            If None, uses module-level CONTOUR_CONFIG. Defaults to None.
    
    Returns:
        Optional[plt.Figure]: Matplotlib Figure object if show=False; None if show=True.
    
    Raises:
        ValueError: If images and passbands have mismatched lengths, or if
            saliency shape does not match images.
        KeyError: If a passband is not found in SDO/AIA colormap registry.
    
    Example:
        >>> images = np.random.rand(7, 512, 512)  # 7 passbands, 512×512 pixels
        >>> passbands = [94, 131, 171, 193, 211, 304, 335]
        >>> fig = plot_aia_image_grid(
        ...     images=images,
        ...     passbands=passbands,
        ...     saliency=attribution_map,
        ...     vmax_percentile=99.9,
        ...     show=False
        ... )
        >>> fig.savefig("output.png", dpi=300, bbox_inches="tight")
    """
    # Input validation
    if images.shape[0] != len(passbands):
        raise ValueError(
            f"images and passbands length mismatch: {images.shape[0]} images "
            f"vs {len(passbands)} passbands"
        )
    
    if saliency is not None and saliency.shape != images.shape:
        raise ValueError(
            f"saliency shape {saliency.shape} must match images shape {images.shape}"
        )
    
    # Use module-level default if no config provided
    if contour_config is None:
        contour_config = CONTOUR_CONFIG
    
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

        # Plot the base image with SDO/AIA colormap
        try:
            aia_cmap = matplotlib.colormaps[f'sdoaia{passband}']
        except KeyError:
            raise KeyError(f"SDO/AIA colormap 'sdoaia{passband}' not found. "
                         f"Available passbands: {sorted(contour_config.keys())}")
        
        if vmax_percentile is not None:
            vmax = np.percentile(image, vmax_percentile)
        else:
            vmax = None
        im = ax.imshow(image, cmap=aia_cmap, origin='lower', vmax=vmax)
        ax.axis("off")

        # Optionally plot contours if saliency is given
        if saliency is not None:
            s = saliency[idx]
            if passband in contour_config:
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
        return None
    else:
        plt.close(fig)
        return fig


# ==================== Main Block ====================
if __name__ == "__main__":
    import argparse
    from pathlib import Path as _Path
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-path",   default="solar_dataset.json")
    parser.add_argument("--model-path",  default="outputs/glad-shape-197/trained_model.pth",
                        help="Path to trained ViT model checkpoint.")
    parser.add_argument("--aarp-id",     type=int, default=3563)
    parser.add_argument("--output-path", default=None,
                        help="Output PNG path. Defaults to "
                             "plots/attribution_contour_grid/<run-id>.png derived from --model-path.")
    parser.add_argument("--stats-file",  default="stats.pkl",
                        help="Path to normalization stats pickle (e.g. stats_raw.pkl for pre-fix models).")
    parser.add_argument("--channels",    type=int, nargs="+", default=None,
                        help="AIA passbands the model was trained on, e.g. --channels 94 131. "
                             "Default: all 7, in wavelength order.")
    parser.add_argument("--model-type",  default="vit", choices=list(VALID_MODEL_TYPES),
                        help="Model architecture: 'vit' (DeepFlare_ViT, default) or "
                             "'vit-pretrained' (torchvision vit_l_16).")
    args = parser.parse_args()

    if args.output_path is None:
        run_id = _Path(args.model_path).parent.name
        args.output_path = f"plots/attribution_contour_grid/{run_id}.png"

    all_passbands = [94, 131, 171, 193, 211, 304, 335]
    channel_indices = None
    passbands = all_passbands
    if args.channels is not None:
        unknown = [c for c in args.channels if c not in all_passbands]
        if unknown:
            parser.error(f"Unknown channel(s) {unknown}. Choices: {all_passbands}")
        channel_indices = [all_passbands.index(c) for c in args.channels]
        passbands = args.channels

    try:
        # Load configuration and data
        config = TrainingConfig(json_path=args.json_path, stats_file=args.stats_file,
                                channel_indices=channel_indices,
                                model_type=VALID_MODEL_TYPES[args.model_type])
        config.trained_model_path = args.model_path
        metadata, model, transform, device = get_data_model(config)
        resize_to = needs_resize(config)
        print(f"Model type: {args.model_type}" + (f"  (resize to {resize_to})" if resize_to else ""))
        training_df, val_df, test_df = dfs_from_metadata(metadata)

        # Select AARP instance and time index
        aarp_id = args.aarp_id
        channel = passbands.index(131) if 131 in passbands else 0
        t_idx = 12

        # Find the AARP in whichever split it belongs to
        aarp_id_df = None
        for df in [test_df, val_df, training_df]:
            if not df.empty and aarp_id in df.aarp_id.values:
                aarp_id_df = df.query(f"aarp_id=={aarp_id}")
                break
        if aarp_id_df is None or aarp_id_df.empty:
            raise ValueError(f"AARP {aarp_id} not found in any split of {args.json_path}")
        s_aarp = single_aarp(aarp_id, aarp_id_df)
        s_images = s_aarp.get_images()
        if channel_indices is not None:
            s_images = s_images[:, channel_indices]
        image = s_images[t_idx]
        
        # Compute attributions
        ig_out = get_attribution_for_image(
            image, s_aarp.label, transform, device, model, ib_size=1, resize_to=resize_to
        )
        saliency = ig_out[channel]
        
        # Plot and save
        print(f"Generating attribution contour grid for AARP {aarp_id} at t_idx={t_idx}...")
        fig = plot_aia_image_grid(
            images=s_images[t_idx],
            passbands=passbands,
            saliency=ig_out,
            contour_config=CONTOUR_CONFIG,  # Use module-level default
            vmax_percentile=DEFAULT_VMAX_PERCENTILE,
            show=False
        )
        
        from pathlib import Path
        Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output_path, bbox_inches="tight", dpi=300)
        print(f"✓ Saved figure to {args.output_path}")
        
    except FileNotFoundError as e:
        print(f"✗ File error: {e}")
    except ValueError as e:
        print(f"✗ Validation error: {e}")
    except KeyError as e:
        print(f"✗ Colormap error: {e}")
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
