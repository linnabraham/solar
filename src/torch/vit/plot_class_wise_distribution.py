import os
import json
from pathlib import Path
from typing import Tuple, Optional, Dict, List
from aarp_ml.dataset import all_wavelengths
import torch
import numpy as np
import matplotlib.pyplot as plt
import scienceplots
import seaborn as sns
from src.torch.vit.ig import single_aarp
from src.torch.vit.utils import dfs_from_metadata, get_metadata_from_json
from src.torch.vit.class_wise_distribution import plot_intensity_distribution

__all__ = [
    "load_images_for_label",
    "plot_distribution_for_passband",
    "plot_all_passbands",
    "configure_plot_style",
    "PASSBAND_CONFIG",
]

# ==================== Module Constants ====================
# Default percentile levels for each AIA passband
PASSBAND_PERCENTILES = {
    94:  [80, 90, 99, 99.9],
    131: [80, 90, 99, 99.9],
    171: [80, 90, 99, 99.9],
    193: [80, 90, 99, 99.9],
    211: [80, 90, 99, 99.9],
    304: [80, 90, 99, 99.9],
    335: [80, 90, 99, 99.9],
}

# Default x-axis (log intensity) ranges for each passband
PASSBAND_X_RANGES = {
    94:  (0, 8),
    131: (0, 6),
    171: (4, 8),
    193: (4, 8),
    211: (3, 8),
    304: (3, 8),
    335: (0, 6),
}

# Combined configuration for easy lookup
PASSBAND_CONFIG = {
    pb: {
        'percentiles': PASSBAND_PERCENTILES.get(pb, [80, 90, 99, 99.9]),
        'x_range': PASSBAND_X_RANGES.get(pb, (0, 8))
    }
    for pb in all_wavelengths
}

DEFAULT_NBINS = 30
DEFAULT_ALPHA = 0.4
DEFAULT_FIGSIZE = (24, 5)
DEFAULT_DPI = 150
DEFAULT_PDF_DPI = 300
DEFAULT_FONT_SIZE = 24

# ==================== Plot Configuration Function ====================
def configure_plot_style(font_size: int = DEFAULT_FONT_SIZE) -> None:
    """Configure matplotlib and seaborn plot styling for distribution plots.
    
    Sets up a consistent visual style using scienceplots (no-latex mode),
    adjusts font sizes, and applies seaborn theming for professional-looking
    distribution plots.
    
    Args:
        font_size (int): Font size for all plot elements. Defaults to 24.
    
    Side effects:
        - Modifies global matplotlib and seaborn settings
        - Uses 'no-latex' style from scienceplots for faster rendering
    """
    plt.style.use(['no-latex'])
    plt.rcParams.update({'font.size': font_size})
    sns.set_theme()

# ==================== Main Functions ====================
def load_images_for_label(df, label: int, verbose: bool = True) -> List[np.ndarray]:
    """Load and stack all AIA images for a specific class label.
    
    Iterates through all unique AARP instances with the given label,
    loads their image sequences, and returns them as a list.
    
    Args:
        df: Pandas DataFrame with columns including 'label', 'aarp_id', and wavelength columns.
        label (int): Class label to filter on (0=non-flare, 1=flare).
        verbose (bool): If True, print AARP ID for each loaded instance. Defaults to True.
    
    Returns:
        List[np.ndarray]: List of image arrays, each of shape [n_timesteps, n_channels, H, W].
    
    Raises:
        ValueError: If label is not 0 or 1.
    """
    if label not in [0, 1]:
        raise ValueError(f"label must be 0 or 1, got {label}")
    
    images_list = []
    for aarp_id in df.query(f'label == {label}').aarp_id.unique():
        if verbose:
            print(aarp_id)
        aarp_id_df = df.query(f'aarp_id == {aarp_id}')
        s_aarp = single_aarp(aarp_id, aarp_id_df)
        s_images = s_aarp.get_images()
        images_list.append(s_images)
    return images_list

def plot_distribution_for_passband(
    passband: int,
    images_list_neg: List[np.ndarray],
    images_list_pos: List[np.ndarray],
    attributions_list_neg: List[torch.Tensor],
    attributions_list_pos: List[torch.Tensor],
    percentile_levels: Optional[List[float]] = None,
    x_range: Optional[Tuple[float, float]] = None,
    nbins: int = DEFAULT_NBINS,
    alpha: float = DEFAULT_ALPHA,
    figsize: Tuple[float, float] = DEFAULT_FIGSIZE,
    dpi: int = DEFAULT_DPI,
) -> Tuple[plt.Figure, np.ndarray]:
    """Generate intensity distribution plot for a single AIA passband.
    
    Creates a multi-panel figure showing intensity distributions of flare vs. non-flare
    regions, stratified by different attribution percentile thresholds. Each panel
    represents a different percentile level used to threshold the attribution maps.
    
    Args:
        passband (int): AIA wavelength (e.g., 94, 131, 171, ...). Must be in all_wavelengths.
        images_list_neg (List[np.ndarray]): List of image arrays for non-flare events.
        images_list_pos (List[np.ndarray]): List of image arrays for flare events.
        attributions_list_neg (List[torch.Tensor]): Attribution tensors for non-flare.
        attributions_list_pos (List[torch.Tensor]): Attribution tensors for flare.
        percentile_levels (Optional[List[float]]): Attribution percentiles to threshold.
            If None, uses PASSBAND_PERCENTILES[passband]. Defaults to None.
        x_range (Optional[Tuple[float, float]]): X-axis range for log(intensity).
            If None, uses PASSBAND_X_RANGES[passband]. Defaults to None.
        nbins (int): Number of histogram bins. Defaults to 30.
        alpha (float): Transparency level for histogram bars [0-1]. Defaults to 0.4.
        figsize (Tuple[float, float]): Figure size (width, height) in inches. Defaults to (24, 5).
        dpi (int): Resolution in dots per inch. Defaults to 150.
    
    Returns:
        Tuple[plt.Figure, np.ndarray]: Matplotlib (figure, axes) tuple.
    
    Raises:
        ValueError: If passband not in PASSBAND_CONFIG.
        KeyError: If internal wavelength lookup fails.
    
    Example:
        >>> fig, axes = plot_distribution_for_passband(
        ...     passband=131,
        ...     images_list_neg=neg_images,
        ...     images_list_pos=pos_images,
        ...     attributions_list_neg=neg_attrs,
        ...     attributions_list_pos=pos_attrs,
        ...     percentile_levels=[80, 90, 99, 99.9]
        ... )
        >>> fig.savefig("passband_131.pdf", bbox_inches="tight")
    """
    # Validate passband
    if passband not in PASSBAND_CONFIG:
        raise ValueError(
            f"passband {passband} not in PASSBAND_CONFIG. "
            f"Valid passbands: {sorted(PASSBAND_CONFIG.keys())}"
        )
    
    # Use defaults if not provided
    if percentile_levels is None:
        percentile_levels = PASSBAND_CONFIG[passband]['percentiles']
    if x_range is None:
        x_range = PASSBAND_CONFIG[passband]['x_range']

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

def plot_all_passbands(
    df,
    images: Tuple[List[np.ndarray], List[np.ndarray]],
    attributions: Tuple[List[torch.Tensor], List[torch.Tensor]],
    passbands: Optional[List[int]] = None,
    save_dir: Optional[str] = None,
    output_format: str = "pdf",
    output_dpi: int = DEFAULT_PDF_DPI,
) -> Dict[int, Tuple[plt.Figure, np.ndarray]]:
    """Generate and optionally save distribution plots for all or specified passbands.
    
    Creates a distribution plot for each passband using default configurations,
    with option to save to disk immediately after generation (freeing memory).
    
    Args:
        df: Pandas DataFrame with passband/wavelength columns for determining defaults.
        images (Tuple[List, List]): Tuple of (images_neg, images_pos) image lists.
        attributions (Tuple[List, List]): Tuple of (attributions_neg, attributions_pos).
        passbands (Optional[List[int]]): List of passbands to plot. If None, uses all_wavelengths.
            Defaults to None.
        save_dir (Optional[str]): Directory to save plots. If None, plots are not saved.
            Defaults to None.
        output_format (str): File format for saved plots ('pdf', 'png', 'jpg', etc.).
            Defaults to 'pdf'.
        output_dpi (int): DPI for saved figures. Defaults to 300.
    
    Returns:
        Dict[int, Tuple]: Dictionary mapping passbands to (figure, axes) tuples.
            If save_dir is provided, figures are closed after saving.
    
    Side effects:
        - If save_dir is not None, creates directory and saves plots to disk
        - Closes figures after saving to free memory
    
    Example:
        >>> plots = plot_all_passbands(
        ...     test_df,
        ...     images=(neg_imgs, pos_imgs),
        ...     attributions=(neg_attrs, pos_attrs),
        ...     passbands=all_wavelengths,
        ...     save_dir="plots/distribution",
        ...     output_format="pdf"
        ... )
        >>> print(f"Generated {len(plots)} plots")
    """
    images_list_neg, images_list_pos = images
    attributions_list_neg, attributions_list_pos = attributions
    
    if passbands is None:
        passbands = list(all_wavelengths)
    
    plots = {}
    for pb in passbands:
        try:
            fig, ax = plot_distribution_for_passband(
                pb,
                images_list_neg, images_list_pos,
                attributions_list_neg, attributions_list_pos
            )
            plots[pb] = (fig, ax)

            # Save immediately if requested
            if save_dir is not None:
                save_path = Path(save_dir)
                save_path.mkdir(parents=True, exist_ok=True)
                output_file = save_path / f"{pb}.{output_format}"
                fig.savefig(
                    str(output_file),
                    format=output_format,
                    dpi=output_dpi,
                    bbox_inches="tight"
                )
                plt.close(fig)
                print(f"✓ Saved {pb} to {output_file}")
        except (ValueError, KeyError) as e:
            print(f"✗ Error processing passband {pb}: {e}")
    
    return plots

# ==================== Main Block ====================
if __name__ == "__main__":
    try:
        # Configure plotting style
        print("Configuring plot style...")
        configure_plot_style(font_size=DEFAULT_FONT_SIZE)
        
        # Load metadata
        print("Loading metadata from solar_dataset.json...")
        metadata = get_metadata_from_json('solar_dataset.json')
        training_df, val_df, test_df = dfs_from_metadata(metadata)
        
        # Load attributions
        print("Loading attributions...")
        attributions_list_neg = torch.load("data/intermediate-outs/attributions_neg.pt")
        attributions_list_pos = torch.load("data/intermediate-outs/attributions_pos.pt")

        # Load images
        print("Loading test images for positive (flare) class...")
        images_list_pos = load_images_for_label(test_df, label=1, verbose=False)
        print("Loading test images for negative (non-flare) class...")
        images_list_neg = load_images_for_label(test_df, label=0, verbose=False)

        images = (images_list_neg, images_list_pos)
        attributions = (attributions_list_neg, attributions_list_pos)

        # Generate and save plots
        print(f"Generating distribution plots for {len(all_wavelengths)} passbands...")
        plots = plot_all_passbands(
            test_df,
            images,
            attributions,
            passbands=list(all_wavelengths),
            save_dir="plots/distribution",
            output_format="pdf",
            output_dpi=300
        )
        
        print(f"✓ Successfully generated {len(plots)} distribution plots")
        
    except FileNotFoundError as e:
        print(f"✗ File not found: {e}")
    except ValueError as e:
        print(f"✗ Validation error: {e}")
    except Exception as e:
        print(f"✗ Unexpected error: {type(e).__name__}: {e}")
