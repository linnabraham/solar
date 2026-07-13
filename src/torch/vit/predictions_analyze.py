"""Prediction analysis and visualization for solar flare ViT model.

Generates GOES X-ray timeseries plots overlaid with model prediction scores
for test/validation AARP samples. Includes optional flare start time markers.

Typical usage:
    python -m src.torch.vit.predictions_analyze
"""

import json
import pandas as pd
from aarp_ml.dataset import all_wavelengths
import torch
from aarp_ml.torch.dataset import AIALogTransform
import pickle
import numpy as np
from torch.utils.data import TensorDataset
from aarp_ml.torch.model import DeepFlare_ViT
import matplotlib.pyplot as plt
import os
from src.torch.vit.ig import single_aarp, make_predictions
import aarp_ml
from astro_utils.utils import get_start_and_end_time
import matplotlib.dates as mdates
from sunpy.timeseries import XRSTimeSeries
import warnings
import gc
from src.data_single import DatasetPaths
from src.torch.vit.utils import get_data_model, dfs_from_metadata
from src.torch.vit.train import TrainingConfig

# ==================== MODULE-LEVEL CONSTANTS ====================

# GOES plotting configuration
DEFAULT_FIGSIZE: tuple = (10, 6)
DEFAULT_DPI: int = 150
OUTPUT_FIGSIZE: tuple = (6, 4)
OUTPUT_DPI: int = 150

# GOES channels and flux limits
GOES_FLUX_MIN: float = 1e-7
GOES_FLUX_MAX: float = 1e-2
FLARE_CLASS_LABELS: list = ['B', 'C', 'M', 'X']
FLARE_CLASS_LOG_RANGE: tuple = (-6.5, -3.5)

# Output paths and filenames
DEFAULT_OUTPUT_HOME: str = "plots/predictions"
PREDICTION_OUTPUT_FILENAME: str = "goes_with_predictions.png"
DEFAULT_TRAINED_MODEL_PATH: str = "outputs/glad-shape-197/trained_model.pth"

# Training parameters
DEFAULT_LEARNING_RATE: float = 0.001

# Prediction visualization
DEFAULT_PREDICTION_ALPHA: float = 0.4

# Annotation and visualization parameters
ANNOTATION_FONTSIZE: int = 10
ANNOTATION_FONTWEIGHT: str = 'bold'
ANNOTATION_COLOR: str = 'blue'
ANNOTATION_ROTATION: int = 90
ANNOTATION_Y_LEVELS: int = 4
ANNOTATION_Y_BASE_OFFSET: float = 0.35
ANNOTATION_ALPHA: float = 0.7
ANNOTATION_Y_OFFSET: int = 2

# GOES observation window
FLARE_TIME_WINDOW_HOURS: float = 24 * 4.5


def plot_goes(
    goes_ts: XRSTimeSeries,
    columns: list = None,
    xlimits: tuple = None,
    ax = None,
    figsize: tuple = DEFAULT_FIGSIZE,
    dpi: int = DEFAULT_DPI,
    **kwargs
):
    """Plot GOES X-ray timeseries data.
    
    Custom implementation with enhanced control over formatting, axis limits,
    and label positioning.
    
    Args:
        goes_ts: sunpy.timeseries.XRSTimeSeries object.
        columns: List of channel names to plot (default: ["xrsa", "xrsb"]).
        xlimits: Tuple of (start, end) timestamps to truncate data.
        ax: Existing Axes object (creates new figure if None).
        figsize: Figure size (width, height) in inches.
        dpi: Dots per inch for figure.
        **kwargs: Passed to ax.plot().
    
    Returns:
        Tuple of (matplotlib Figure, Axes).
    """
    plot_settings = {
        "xrsa": ["blue", r"0.5$-$4.0 $\mathrm{\AA}$"],
        "xrsb": ["red", r"1.0$-$8.0 $\mathrm{\AA}$"]
    }
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        fig = ax.figure
    
    if columns is None:
        columns = ["xrsa", "xrsb"]
    
    if xlimits:
        a, b = xlimits
        data = goes_ts.truncate(a, b).data
    else:
        data = goes_ts.data
    
    for channel in columns:
        ax.plot(
            data.index, data[channel], "-", 
            label=plot_settings[channel][1],
            color=plot_settings[channel][0], lw=1, **kwargs
        )
    
    ax.set_yscale("log")
    ax.set_ylim(GOES_FLUX_MIN, GOES_FLUX_MAX)
    ax.set_ylabel("Watts m$^{-2}$")

    locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

    ax.tick_params(axis='x', rotation=45)
    centers = np.logspace(*FLARE_CLASS_LOG_RANGE, len(FLARE_CLASS_LABELS))

    for value, label in zip(centers, FLARE_CLASS_LABELS):
        ax.text(-0.02, value, label, transform=ax.get_yaxis_transform(), horizontalalignment='center')
    
    ax.yaxis.grid(True, "major")
    ax.xaxis.grid(False, "major")
    ax.legend()
    return fig, ax



def plot_custom_goes_with_aarp_sampling(
    goes_ts: XRSTimeSeries,
    timestamps,
    columns: list = None,
    xlimits: tuple = None,
    ax = None,
    figsize: tuple = DEFAULT_FIGSIZE,
    dpi: int = DEFAULT_DPI,
    **kwargs
):
    """Plot GOES data with AARP sampling timestamps marked.
    
    Overlays vertical lines for each sample timestamp on the GOES plot,
    allowing visualization of when the model was observing data.
    
    Args:
        goes_ts: sunpy.timeseries.XRSTimeSeries object.
        timestamps: DatetimeIndex of sampling times to mark.
        columns: List of channel names to plot.
        xlimits: Tuple of (start, end) timestamps.
        ax: Existing Axes object (creates new if None).
        figsize: Figure size (width, height) in inches.
        dpi: Dots per inch for figure.
        **kwargs: Passed to plot_goes().
    
    Returns:
        Tuple of (matplotlib Figure, Axes).
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        fig = ax.figure

    fig, ax = plot_goes(goes_ts, columns=columns, xlimits=xlimits, ax=ax, figsize=figsize, dpi=dpi, **kwargs)
    if xlimits:
        xlim_start = pd.Timestamp(xlimits[0], tz="UTC")
        xlim_end = pd.Timestamp(xlimits[1], tz="UTC")
        timestamps = timestamps[timestamps.between(xlim_start, xlim_end)]
    for ts in timestamps:
        ax.axvline(ts, color='grey', linestyle='--')
    return fig, ax

def vizualize_goes_ts_predictions(
    goes_ts: XRSTimeSeries,
    s_aarp,
    predicted_scores,
    goes_event_list_df = None,
    flare_start: bool = False,
    columns: list = None,
    xlimits: tuple = None,
    resample: bool = False,
    figsize: tuple = DEFAULT_FIGSIZE,
    dpi: int = DEFAULT_DPI,
    **kwargs
):
    """Overlay model prediction scores on GOES timeseries.
    
    Creates dual-axis plot with GOES X-ray flux (primary y-axis)
    and model prediction scores (secondary y-axis). Optionally marks
    flare start time.
    
    Args:
        goes_ts: sunpy.timeseries.XRSTimeSeries object.
        s_aarp: single_aarp instance with timestamps and aarp_id.
        predicted_scores: 1D tensor/array of prediction scores (0-1).
        goes_event_list_df: DataFrame with flare event metadata.
        flare_start: Mark flare start time if True.
        columns: List of channel names to plot (default: ["xrsb"]).
        xlimits: Tuple of (start, end) timestamps.
        resample: Resample GOES to match AARP timestamps if True.
        figsize: Figure size (width, height) in inches.
        dpi: Dots per inch for figure.
        **kwargs: Passed to plot_goes() (e.g., alpha).
    
    Returns:
        Tuple of (matplotlib Figure, Axes).
    
    Raises:
        ValueError: If flare_start=True but goes_event_list_df is None.
    """
    if columns is None:
        columns = ["xrsb"]
    
    if resample:
        goes_ts = XRSTimeSeries(
            data=goes_ts.data.reindex(
                pd.DatetimeIndex(s_aarp.timestamps),
                method="nearest",
                tolerance=pd.Timedelta(seconds=2)
            ),
            meta=goes_ts.meta
        )

    fig, ax = plot_goes(goes_ts, columns=['xrsb'], xlimits=xlimits, **kwargs)

    if flare_start:
        if goes_event_list_df is None:
            raise ValueError("goes_event_list should be given if flare_start is True")
        result_event_list = goes_event_list_df.query(f"harpnum == {s_aarp.aarp_id}")
        if len(result_event_list) != 1:
            warnings.warn(f"Expected 1 event, got {len(result_event_list)}")
        f_st = result_event_list[['start_time', 'peak_time', 'end_time']].values[0][0]
        ax.axvline(f_st, color='black', linestyle='--', label='flare start')
        ax.legend()

    # xlim_start = pd.Timestamp(xlimits[0], tz="UTC")
    # xlim_end = pd.Timestamp(xlimits[1], tz="UTC")
    xlim_start = pd.Timestamp(xlimits[0])
    xlim_end = pd.Timestamp(xlimits[1])
    mask = s_aarp.timestamps.between(xlim_start, xlim_end)

    # If predicted_scores is a NumPy array:
    sliced_scores = predicted_scores[mask.to_numpy()]
    filtered_ts = s_aarp.timestamps[s_aarp.timestamps.between(xlim_start, xlim_end)]
    ax2 = ax.twinx()
    # ax2.scatter(s_aarp.timestamps.tolist(), predicted_scores.numpy(), color='brown', linestyle='-', linewidth=1, s=1, label='Predicted Scores')'
    ax2.scatter(filtered_ts, sliced_scores, color='blue', linestyle='-', linewidth=1, s=1, label='Predicted Scores')
    ax2.tick_params(axis='y', pad=15)  
    ax2.set_ylim(-0.1, 1.1)
    ax2.legend()
    plt.title(f"ViT prediction scores for AARP {s_aarp.aarp_id} overlaid on GOES X-ray timeseries", fontsize=9)
    return fig, ax


def get_aarp_seq_dataset(s_aarp, transform, device, channel_indices=None):
    """Load AARP image sequence and return as TensorDataset.
    
    Args:
        s_aarp: single_aarp instance with image loading methods.
        transform: AIALogTransform normalization transform.
        device: torch.device (cuda or cpu).
    
    Returns:
        torch.utils.data.TensorDataset with normalized images.
    """
    s_images = s_aarp.get_images()  # Shape: [N, 7, 512, 512]
    if channel_indices is not None:
        s_images = s_images[:, channel_indices]
    tensor_images = torch.from_numpy(s_images).to(torch.float32).to(device)
    tensor_data = transform(tensor_images)
    dataset = TensorDataset(tensor_data)
    return dataset

def make_prediction_plot(aarp_id, metadata_df, transform, model, device, output_home, resume=False, channel_indices=None):
    """Generate and save prediction plot for a single AARP sample.
    
    Loads AARP image sequence, generates model predictions, fetches GOES
    timeseries data, and creates output plot with predictions overlaid.
    Optionally marks flare start time.
    
    Args:
        aarp_id: AARP region identifier.
        metadata_df: DataFrame with sample metadata.
        transform: AIALogTransform normalization.
        model: Trained ViT model.
        device: torch.device (cuda or cpu).
        output_home: Root directory for output plots.
        resume: Skip if output exists (default: False).
    
    Returns:
        None
    """
    output_dir = f"{output_home}/{aarp_id}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        if resume:
            print(f"Output directory {output_dir} already exists, skipping {aarp_id}")
            return None

    print(f"Using {aarp_id=}")

    aarp_id_df = metadata_df.query(f'aarp_id == {aarp_id}')
    s_aarp = single_aarp(aarp_id, aarp_id_df)

    dataset = get_aarp_seq_dataset(s_aarp, transform, device, channel_indices=channel_indices)
    predictions = make_predictions(dataset, model=model, device=device)
    torch.cuda.empty_cache()
    gc.collect()
    goes_event_list_path = DatasetPaths(parent_dir=".").goes_event_with_aarp
    print(f"{goes_event_list_path}")
    goes_event_list = pd.read_csv(goes_event_list_path, parse_dates=["event_date", "start_time", "peak_time","end_time"])

    softmax_predictions = torch.softmax(predictions, dim=1)
    predicted_scores, predicted_labels = torch.max(softmax_predictions , dim=1)

    # plt.plot(predicted_scores)
    # plt.savefig(f"{output_dir}/predicted_scores.png", bbox_inches="tight", dpi=150)
    # plt.close()

    # Make plots combining data with predictions
    print("Half width duration of observations", (s_aarp.timestamps.max() - s_aarp.timestamps.min())/2)
    fl_start, fl_end = get_start_and_end_time(s_aarp.get_midtime(), FLARE_TIME_WINDOW_HOURS * 60)
    goes_ts = aarp_ml.utils.run_fetch_goes(fl_start, fl_end)

    matched_events = goes_event_list.query(f"harpnum == {aarp_id}")
    if not matched_events.empty:
        f_st = matched_events[['start_time', 'peak_time', 'end_time']].values[0][0]
        print("Start of the flare", f_st)
        flare_start = True
    else:
        flare_start = False

    xlimits = (s_aarp.timestamps.min(), s_aarp.timestamps.max())
    fig, ax = vizualize_goes_ts_predictions(
        goes_ts, s_aarp, predicted_scores, 
        goes_event_list_df=goes_event_list,
        flare_start=flare_start, xlimits=xlimits,
        resample=False, figsize=OUTPUT_FIGSIZE, dpi=OUTPUT_DPI,
        alpha=DEFAULT_PREDICTION_ALPHA
    )
    fig.savefig(f"{output_dir}/{PREDICTION_OUTPUT_FILENAME}", bbox_inches="tight", dpi=OUTPUT_DPI)
    plt.close(fig)


def main():
    """Main execution: generate prediction plots for test/validation samples.

    Loads model and data, processes each sample in test/validation split,
    generates GOES plots with overlaid predictions, and saves output.
    """
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-path",  default="solar_dataset.json")
    parser.add_argument("--model-path", default=DEFAULT_TRAINED_MODEL_PATH,
                        help="Path to trained ViT model checkpoint.")
    parser.add_argument("--output-dir", default=None,
                        help="Root directory for output plots. Defaults to "
                             "plots/predictions/<run-id> derived from --model-path.")
    parser.add_argument("--stats-file", default="stats.pkl",
                        help="Path to normalization stats pickle (e.g. stats_raw.pkl for pre-fix models).")
    parser.add_argument("--channels", type=int, nargs="+", default=None,
                        help="AIA passbands the model was trained on, e.g. --channels 94 131. "
                             "Default: all 7, in wavelength order.")
    args = parser.parse_args()

    if args.output_dir is None:
        run_id = Path(args.model_path).parent.name
        args.output_dir = f"plots/predictions/{run_id}"

    channel_indices = None
    if args.channels is not None:
        from aarp_ml.dataset import all_wavelengths
        unknown = [c for c in args.channels if c not in all_wavelengths]
        if unknown:
            parser.error(f"Unknown channel(s) {unknown}. Choices: {all_wavelengths}")
        channel_indices = [all_wavelengths.index(c) for c in args.channels]

    config = TrainingConfig(json_path=args.json_path, stats_file=args.stats_file,
                            channel_indices=channel_indices)
    config.trained_model_path = args.model_path
    metadata, model, transform, device = get_data_model(config)
    training_df, val_df, test_df = dfs_from_metadata(metadata)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(args.model_path, map_location=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=DEFAULT_LEARNING_RATE)
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', -1) + 1
    else:
        model.load_state_dict(checkpoint)
    model = model.to(device)

    output_home = args.output_dir
    os.makedirs(output_home, exist_ok=True)

    for aarp_id in test_df.aarp_id.unique().tolist() if not test_df.empty else []:
        make_prediction_plot(aarp_id, test_df, transform, model, device, output_home,
                             channel_indices=channel_indices)
    for aarp_id in val_df.aarp_id.unique().tolist() if not val_df.empty else []:
        make_prediction_plot(aarp_id, val_df, transform, model, device, output_home,
                             channel_indices=channel_indices)

if __name__ == "__main__":
    main()

