"""Prediction analysis and visualization for solar flare ViT model.

Generates GOES X-ray timeseries plots overlaid with model prediction scores
for test/validation AARP samples. Includes optional flare start time markers
and sampling time annotations.

Typical usage:
    python -m src.torch.vit.predictions_analyze
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import warnings

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
import matplotlib.dates as mdates
import os
from src.torch.vit.ig import single_aarp, make_predictions
import aarp_ml
from astro_utils.utils import get_start_and_end_time
from sunpy.timeseries import XRSTimeSeries
import gc
from src.data_single import DatasetPaths
from src.torch.vit.utils import get_data_model, dfs_from_metadata
from src.torch.vit.train import TrainingConfig

# ==================== MODULE-LEVEL CONSTANTS ====================

# GOES Plotting Configuration
DEFAULT_FIGSIZE: Tuple[float, float] = (10, 6)
DEFAULT_DPI: int = 150
DEFAULT_OUTPUT_FIGSIZE: Tuple[float, float] = (6, 4)
DEFAULT_OUTPUT_DPI: int = 150

# GOES channels with labels
GOES_CHANNELS: Dict[str, List[str]] = {
    "xrsa": ["blue", r"0.5$-$4.0 $\mathrm{\AA}$"],
    "xrsb": ["red", r"1.0$-$8.0 $\mathrm{\AA}$"]
}
DEFAULT_CHANNELS: List[str] = ["xrsa", "xrsb"]
DEFAULT_PLOT_CHANNEL: str = "xrsb"

# GOES flux limits and class boundaries
GOES_FLUX_MIN: float = 1e-7
GOES_FLUX_MAX: float = 1e-2
FLARE_CLASS_LABELS: List[str] = ['B', 'C', 'M', 'X']
FLARE_CLASS_LOG_CENTERS: Tuple[float, float] = (-6.5, -3.5)

# Prediction visualization
DEFAULT_PREDICTION_SCATTER_SIZE: int = 1
DEFAULT_PREDICTION_ALPHA: float = 0.4
DEFAULT_PREDICTION_COLOR: str = 'blue'
PREDICTION_SCORE_LABEL: str = 'Predicted Scores'

# Annotation styling
ANNOTATION_FONTSIZE: int = 10
ANNOTATION_FONTWEIGHT: str = 'bold'
ANNOTATION_COLOR: str = 'blue'
ANNOTATION_Y_LEVELS: int = 4
ANNOTATION_Y_BASE_OFFSET: float = 0.35
ANNOTATION_ALPHA: float = 0.7

# Flare event styling
FLARE_START_COLOR: str = 'black'
FLARE_START_LINESTYLE: str = '--'
FLARE_START_LABEL: str = 'flare start'

# Grid styling
AARP_SAMPLING_LINESTYLE: str = '--'
AARP_SAMPLING_COLOR: str = 'grey'

# Output configuration
DEFAULT_OUTPUT_DIR: str = "plots/predictions"
DEFAULT_RESUME: bool = False
PREDICTION_OUTPUT_FILENAME: str = "goes_with_predictions.png"

# Prediction shape configuration
PREDICTION_SCORE_Y_MIN: float = -0.1
PREDICTION_SCORE_Y_MAX: float = 1.1


@dataclass
class PredictionConfig:
    """Configuration for prediction analysis and visualization.
    
    Attributes:
        json_path: Path to solar dataset JSON file.
        stats_file: Path to pickled statistics.
        trained_model_path: Path to trained ViT checkpoint.
        output_dir: Root directory for prediction output (plots/predictions).
        resume: Skip existing output directories if True.
        figsize: Figure size for output plots.
        dpi: DPI for output plots.
        goes_channels: Dictionary mapping channel names to [color, label].
        flare_time_window_hours: Half-width of GOES observation window in hours.
    
    Raises:
        FileNotFoundError: If json_path or stats_file do not exist.
        ValueError: If figsize dimensions invalid.
    """
    json_path: str = "solar_dataset.json"
    stats_file: str = "stats.pkl"
    trained_model_path: str = "outputs/glad-shape-197/trained_model.pth"
    output_dir: str = DEFAULT_OUTPUT_DIR
    resume: bool = DEFAULT_RESUME
    figsize: Tuple[float, float] = DEFAULT_OUTPUT_FIGSIZE
    dpi: int = DEFAULT_OUTPUT_DPI
    goes_channels: Dict[str, List[str]] = field(default_factory=lambda: GOES_CHANNELS.copy())
    flare_time_window_hours: float = 4.5 * 24  # 4.5 days
    
    def __post_init__(self) -> None:
        """Validate configuration."""
        if not os.path.exists(self.json_path):
            raise FileNotFoundError(f"JSON file not found: {self.json_path}")
        if not os.path.exists(self.stats_file):
            raise FileNotFoundError(f"Stats file not found: {self.stats_file}")
        if self.figsize[0] <= 0 or self.figsize[1] <= 0:
            raise ValueError(f"Invalid figsize: {self.figsize}")
        if self.dpi < 50:
            raise ValueError(f"DPI must be >= 50, got {self.dpi}")
        os.makedirs(self.output_dir, exist_ok=True)


def plot_goes(
    goes_ts: XRSTimeSeries,
    columns: Optional[List[str]] = None,
    xlimits: Optional[Tuple] = None,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = DEFAULT_FIGSIZE,
    dpi: int = DEFAULT_DPI,
    **kwargs
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot GOES X-ray timeseries data.
    
    Custom implementation replacing XRSTimeSeries.plot with enhanced control
    over formatting, axis limits, and label positioning.
    
    Args:
        goes_ts: sunpy.timeseries.XRSTimeSeries object.
        columns: List of channel names to plot (default: all available).
        xlimits: Tuple of (start, end) timestamps to truncate data.
        ax: Existing Axes object (creates new figure if None).
        figsize: Figure size (width, height) in inches.
        dpi: Dots per inch for figure.
        **kwargs: Passed to ax.plot().
    
    Returns:
        Tuple of (matplotlib Figure, Axes).
    
    Raises:
        ValueError: If invalid column names provided.
        KeyError: If goes_ts missing expected data structure.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        fig = ax.figure
    
    if columns is None:
        columns = DEFAULT_CHANNELS
    
    # Validate columns
    invalid_cols = [c for c in columns if c not in GOES_CHANNELS]
    if invalid_cols:
        raise ValueError(f"Invalid GOES channels: {invalid_cols}")
    
    # Truncate data if xlimits provided
    if xlimits:
        a, b = xlimits
        data = goes_ts.truncate(a, b).data
    else:
        data = goes_ts.data
    
    # Plot each channel
    for channel in columns:
        color, label = GOES_CHANNELS[channel]
        ax.plot(
            data.index, data[channel],
            "-", label=label, color=color, lw=1, **kwargs
        )
    
    # Set y-axis scaling and limits
    ax.set_yscale("log")
    ax.set_ylim(GOES_FLUX_MIN, GOES_FLUX_MAX)
    ax.set_ylabel("Watts m$^{-2}$")
    
    # Configure date/time axis
    locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    ax.tick_params(axis='x', rotation=45)
    
    # Add flare class labels (B, C, M, X)
    labels = FLARE_CLASS_LABELS
    centers = np.logspace(*FLARE_CLASS_LOG_CENTERS, len(labels))
    for value, label in zip(centers, labels):
        ax.text(
            -0.02, value, label,
            transform=ax.get_yaxis_transform(),
            horizontalalignment='center'
        )
    
    # Grid configuration
    ax.yaxis.grid(True, "major")
    ax.xaxis.grid(False, "major")
    ax.legend()
    
    return fig, ax


def plot_goes_with_sampling(
    goes_ts: XRSTimeSeries,
    timestamps: pd.DatetimeIndex,
    columns: Optional[List[str]] = None,
    xlimits: Optional[Tuple] = None,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = DEFAULT_FIGSIZE,
    dpi: int = DEFAULT_DPI,
    **kwargs
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot GOES timeseries with AARP sampling timestamps marked.
    
    Wrapper around plot_goes() that overlays vertical lines for each sample
    timestamp, allowing visualization of when the model was observing data
    relative to the X-ray event.
    
    Args:
        goes_ts: sunpy.timeseries.XRSTimeSeries object.
        timestamps: DatetimeIndex of sampling times to mark with vertical lines.
        columns: List of channel names to plot (default: xrsb only).
        xlimits: Tuple of (start, end) timestamps to limit x-axis.
        ax: Existing Axes object (creates new figure if None).
        figsize: Figure size (width, height) in inches.
        dpi: Dots per inch for figure.
        **kwargs: Passed to plot_goes().
    
    Returns:
        Tuple of (matplotlib Figure, Axes) with sampling times marked.
    
    Raises:
        ValueError: If timestamps empty or invalid format.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        fig = ax.figure
    
    # Plot GOES data
    fig, ax = plot_goes(
        goes_ts, columns=columns or [DEFAULT_PLOT_CHANNEL],
        xlimits=xlimits, ax=ax, figsize=figsize, dpi=dpi, **kwargs
    )
    
    # Mark sampling times
    if xlimits:
        xlim_start = pd.Timestamp(xlimits[0], tz="UTC")
        xlim_end = pd.Timestamp(xlimits[1], tz="UTC")
        timestamps = timestamps[timestamps.between(xlim_start, xlim_end)]
    
    for ts in timestamps:
        ax.axvline(ts, color=AARP_SAMPLING_COLOR, linestyle=AARP_SAMPLING_LINESTYLE)
    
    return fig, ax


def visualize_predictions_on_goes(
    goes_ts: XRSTimeSeries,
    s_aarp: single_aarp,
    predicted_scores: torch.Tensor,
    goes_event_list_df: Optional[pd.DataFrame] = None,
    flare_start: Optional[bool] = False,
    columns: Optional[List[str]] = None,
    xlimits: Optional[Tuple] = None,
    figsize: Tuple[float, float] = DEFAULT_FIGSIZE,
    dpi: int = DEFAULT_DPI,
    **kwargs
) -> Tuple[plt.Figure, plt.Axes]:
    """Overlay model prediction scores on GOES timeseries.
    
    Creates dual-axis plot with GOES X-ray flux on primary y-axis and
    model prediction scores on secondary y-axis. Optionally marks flare
    start time.
    
    Args:
        goes_ts: sunpy.timeseries.XRSTimeSeries object.
        s_aarp: single_aarp instance with timestamps and aarp_id.
        predicted_scores: 1D torch.Tensor of prediction scores (0-1).
        goes_event_list_df: DataFrame with flare event metadata.
        flare_start: Mark flare start time if True (requires goes_event_list_df).
        columns: List of channel names (default: xrsb only).
        xlimits: Tuple of (start, end) timestamps.
        figsize: Figure size (width, height) in inches.
        dpi: Dots per inch for figure.
        **kwargs: Passed to plot_goes().
    
    Returns:
        Tuple of (matplotlib Figure, primary Axes).
    
    Raises:
        ValueError: If flare_start=True but goes_event_list_df None.
        IndexError: If scores/timestamps length mismatch.
    """
    if columns is None:
        columns = [DEFAULT_PLOT_CHANNEL]
    
    # Plot GOES data
    fig, ax = plot_goes(
        goes_ts, columns=columns, xlimits=xlimits,
        figsize=figsize, dpi=dpi, **kwargs
    )
    
    # Mark flare start if requested
    if flare_start:
        if goes_event_list_df is None:
            raise ValueError("goes_event_list_df required when flare_start=True")
        
        result_event_list = goes_event_list_df.query(f"harpnum == {s_aarp.aarp_id}")
        if len(result_event_list) != 1:
            warnings.warn(
                f"Expected 1 flare event for AARP {s_aarp.aarp_id}, "
                f"got {len(result_event_list)}"
            )
        if len(result_event_list) > 0:
            f_st = result_event_list[['start_time', 'peak_time', 'end_time']].values[0][0]
            ax.axvline(
                f_st, color=FLARE_START_COLOR, linestyle=FLARE_START_LINESTYLE,
                label=FLARE_START_LABEL
            )
            ax.legend()
    
    # Prepare time window for masking
    xlim_start = pd.Timestamp(xlimits[0])
    xlim_end = pd.Timestamp(xlimits[1])
    mask = s_aarp.timestamps.between(xlim_start, xlim_end)
    
    # Extract scores and times in window
    if isinstance(predicted_scores, torch.Tensor):
        sliced_scores = predicted_scores[mask.to_numpy()].numpy()
    else:
        sliced_scores = predicted_scores[mask.to_numpy()]
    
    filtered_ts = s_aarp.timestamps[mask]
    
    # Create secondary y-axis for predictions
    ax2 = ax.twinx()
    ax2.scatter(
        filtered_ts, sliced_scores,
        color=DEFAULT_PREDICTION_COLOR, linestyle='-',
        linewidth=1, s=DEFAULT_PREDICTION_SCATTER_SIZE,
        label=PREDICTION_SCORE_LABEL, alpha=kwargs.get('alpha', DEFAULT_PREDICTION_ALPHA)
    )
    ax2.set_ylim(PREDICTION_SCORE_Y_MIN, PREDICTION_SCORE_Y_MAX)
    ax2.tick_params(axis='y', pad=15)
    ax2.legend()
    
    plt.title(
        f"ViT prediction scores for AARP {s_aarp.aarp_id} overlaid on GOES X-ray timeseries",
        fontsize=9
    )
    
    return fig, ax


def get_aarp_sequence_dataset(
    s_aarp: single_aarp,
    transform: AIALogTransform,
    device: torch.device
) -> TensorDataset:
    """Load image sequence for AARP and return as TensorDataset.
    
    Retrieves all images for AARP ID, applies transformation, and creates
    a PyTorch TensorDataset for batch inference.
    
    Args:
        s_aarp: single_aarp instance with image loading methods.
        transform: AIALogTransform normalization transform.
        device: torch.device (cuda or cpu).
    
    Returns:
        torch.utils.data.TensorDataset containing normalized images.
    
    Raises:
        RuntimeError: If image loading or transformation fails.
    """
    s_images = s_aarp.get_images()  # Shape: [N, 7, 512, 512]
    tensor_images = torch.from_numpy(s_images).to(torch.float32)
    tensor_data = transform(tensor_images)
    tensor_data = tensor_data.to(device)
    dataset = TensorDataset(tensor_data)
    return dataset


def make_prediction_plot(
    aarp_id: int,
    metadata_df: pd.DataFrame,
    transform: AIALogTransform,
    model: torch.nn.Module,
    device: torch.device,
    output_dir: str,
    resume: bool = False
) -> Optional[str]:
    """Generate and save prediction plot for a single AARP sample.
    
    Loads AARP image sequence, generates model predictions, fetches GOES
    timeseries data, and creates an output plot with predictions overlaid
    on GOES X-ray flux. Optionally marks the flare start time.
    
    Args:
        aarp_id: AARP region identifier.
        metadata_df: DataFrame with sample metadata (aarp_id, timestamps, etc.).
        transform: AIALogTransform normalization.
        model: Trained ViT model in eval mode.
        device: torch.device for inference.
        output_dir: Root directory for saving output plots (plots/predictions).
        resume: Skip AARP if output directory exists (default: False).
    
    Returns:
        Output directory path if successful, None if skipped.
    
    Raises:
        FileNotFoundError: If GOES event list not found.
        RuntimeError: If model inference or GOES fetch fails.
        IOError: If file write fails.
    """
    aarp_output_dir = f"{output_dir}/{aarp_id}"
    
    # Check resume mode
    if os.path.exists(aarp_output_dir):
        if resume:
            print(f"✓ Skipping {aarp_id} (output exists)")
            return None
        # If not resuming, remove existing directory to regenerate
    
    os.makedirs(aarp_output_dir, exist_ok=True)
    print(f"Processing AARP {aarp_id}")
    
    # Load AARP data and make predictions
    aarp_id_df = metadata_df.query(f'aarp_id == {aarp_id}')
    s_aarp = single_aarp(aarp_id, aarp_id_df)
    dataset = get_aarp_sequence_dataset(s_aarp, transform, device)
    predictions = make_predictions(dataset, model=model, device=device)
    
    # Free GPU memory
    torch.cuda.empty_cache()
    gc.collect()
    
    # Load GOES event list
    goes_event_list_path = DatasetPaths(parent_dir=".").goes_event_with_aarp
    goes_event_list = pd.read_csv(
        goes_event_list_path,
        parse_dates=["event_date", "start_time", "peak_time", "end_time"]
    )
    
    # Compute prediction scores from logits
    softmax_predictions = torch.softmax(predictions, dim=1)
    predicted_scores, predicted_labels = torch.max(softmax_predictions, dim=1)
    
    # Fetch GOES timeseries data for observation window
    fl_start, fl_end = get_start_and_end_time(
        s_aarp.get_midtime(),
        s_aarp.timestamps.max() - s_aarp.timestamps.min() / 2
    )
    goes_ts = aarp_ml.utils.run_fetch_goes(fl_start, fl_end)
    
    # Check if flare event exists for this AARP
    matched_events = goes_event_list.query(f"harpnum == {aarp_id}")
    flare_start = not matched_events.empty
    
    if flare_start:
        print(f"  Flare start: {matched_events[['start_time']].values[0][0]}")
    
    # Create and save plot
    xlimits = (s_aarp.timestamps.min(), s_aarp.timestamps.max())
    fig, ax = visualize_predictions_on_goes(
        goes_ts, s_aarp, predicted_scores,
        goes_event_list_df=goes_event_list,
        flare_start=flare_start,
        xlimits=xlimits,
        resample=False,
        figsize=DEFAULT_OUTPUT_FIGSIZE,
        dpi=DEFAULT_OUTPUT_DPI
    )
    
    output_path = f"{aarp_output_dir}/{PREDICTION_OUTPUT_FILENAME}"
    fig.savefig(output_path, bbox_inches="tight", dpi=DEFAULT_OUTPUT_DPI)
    plt.close(fig)
    print(f"  ✓ Saved: {output_path}")
    
    return aarp_output_dir


def main() -> None:
    """Main execution: generate prediction plots for test/validation samples.
    
    Loads model and data, processes each sample in test/validation split,
    generates GOES plots with overlaid predictions, and saves to plots/predictions/
    directory. Optionally uses cached GOES data (resample=False).
    
    Raises:
        FileNotFoundError: If required files not found.
        RuntimeError: If model loading or inference fails.
    """
    # Load configuration and initialize
    config = PredictionConfig()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Device: {device}")
    print(f"Output directory: {config.output_dir}")
    
    # Load model and data
    metadata, model, transform, _ = get_data_model(
        TrainingConfig(
            json_path=config.json_path,
            stats_file=config.stats_file,
            trained_model_path=config.trained_model_path
        )
    )
    training_df, val_df, test_df = dfs_from_metadata(metadata)
    
    # Process test samples
    print("\nProcessing test samples:")
    test_aarp_ids = test_df.aarp_id.unique().tolist()
    for i, aarp_id in enumerate(test_aarp_ids, 1):
        try:
            make_prediction_plot(
                aarp_id, test_df, transform, model, device,
                config.output_dir, resume=config.resume
            )
        except Exception as e:
            print(f"  ✗ Error processing AARP {aarp_id}: {e}")
            continue
    
    # Process validation samples
    print("\nProcessing validation samples:")
    val_aarp_ids = val_df.aarp_id.unique().tolist()
    for i, aarp_id in enumerate(val_aarp_ids, 1):
        try:
            make_prediction_plot(
                aarp_id, val_df, transform, model, device,
                config.output_dir, resume=config.resume
            )
        except Exception as e:
            print(f"  ✗ Error processing AARP {aarp_id}: {e}")
            continue
    
    print(f"\n✓ Prediction plots complete. Results saved to {config.output_dir}/")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, RuntimeError, ValueError) as e:
        print(f"✗ Fatal error: {e}")
        raise

