"""Raw intensity and IG attribution quantile timeseries for AARP sequences.

Generates a per-AARP multi-panel figure:
  Row 0    — GOES X-ray flux with flare peak marker (if positive class)
  Rows 1-7 — per AIA passband:
      left  : log(raw intensity) — median ± IQR shaded band + 99th percentile line
      right : IG attribution upper tail — 95th and 99th percentile lines only
               (median ≈ 0 for sparse IG maps and carries no information)

Outputs:
    plots/intensity_boxplots/{aarp_id}/raw_and_attribution_timeseries.png
"""

import os
import gc
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import aarp_ml

from aarp_ml.dataset import all_wavelengths
from astro_utils.utils import get_start_and_end_time
from torch.utils.data import TensorDataset
from src.torch.vit.ig import single_aarp, make_predictions
from src.torch.vit.class_wise_distribution import run_pred_and_ig
from src.torch.vit.predictions_analyze import plot_goes, FLARE_TIME_WINDOW_HOURS
from src.torch.vit.utils import get_data_model, dfs_from_metadata, needs_resize
from src.torch.vit.train import TrainingConfig
from src.data_single import DatasetPaths
from torchvision.transforms import v2

TRAINED_MODEL_PATH: str = "outputs/glad-shape-197/trained_model.pth"
OUTPUT_HOME: str = "plots/intensity_boxplots"
FIGSIZE: tuple = (16, 26)
OUTPUT_DPI: int = 150
N_XTICK_LABELS: int = 8
VALID_MODEL_TYPES = {"vit": "deepflare_vit", "vit-pretrained": "vit_pretrained"}

RAW_PERCENTILES: list = [25, 50, 75, 99]
ATTR_PERCENTILES: list = [95, 99]


def _spatial_percentiles(data_3d, percentiles):
    """Compute per-timestep percentiles over the spatial dimensions.

    Uses nanpercentile so that NaN pixels (dead detector pixels, FITS artifacts)
    are ignored rather than propagating to the entire timestep's value.

    Args:
        data_3d: numpy array [T, H, W]
        percentiles: list of integer percentile values

    Returns:
        dict mapping each percentile value to a [T] numpy array
    """
    return {p: np.nanpercentile(data_3d, p, axis=(1, 2)) for p in percentiles}


def _set_time_xticks(ax, timestamps, n_ticks=N_XTICK_LABELS):
    """Place evenly spaced datetime labels on the integer x-axis."""
    T = len(timestamps)
    positions = np.linspace(0, T - 1, n_ticks, dtype=int)
    ax.set_xticks(positions)
    ax.set_xticklabels(
        [pd.Timestamp(timestamps.iloc[i]).strftime("%m-%d\n%H:%M") for i in positions],
        fontsize=6,
    )


def _flare_index(timestamps, flare_peak_time):
    """Return the integer timestep index closest to flare_peak_time."""
    ts = pd.to_datetime(timestamps.values)
    peak = pd.Timestamp(flare_peak_time)
    if ts.tz is not None and peak.tzinfo is None:
        ts = ts.tz_localize(None)
    elif ts.tz is None and peak.tzinfo is not None:
        peak = peak.tz_localize(None)
    return int(np.argmin(np.abs(ts - peak)))


def plot_intensity_and_attribution_timeseries(
    aarp_id,
    s_images,
    attributions,
    timestamps,
    goes_ts,
    flare_peak_time,
    output_dir,
    channels=None,
    pred_scores=None,
    figsize=FIGSIZE,
    dpi=OUTPUT_DPI,
):
    """Build and save the combined quantile timeseries figure for one AARP.

    Args:
        aarp_id: Integer AARP identifier.
        s_images: numpy array [T, 7, H, W] of raw FITS pixel values.
        attributions: list of T numpy arrays each shaped [7, H, W] from IG.
        timestamps: pandas Series of T timestamps aligned with s_images rows.
        goes_ts: sunpy XRSTimeSeries covering the observation window.
        flare_peak_time: datetime-like peak time, or None for non-flare events.
        output_dir: Directory to write the output PNG into.
        channels: List of AIA wavelengths to include (default: all 7).
        pred_scores: numpy array [T] of per-timestep flare softmax probabilities.
            When provided, overlaid on the GOES panel as a secondary y-axis so
            attribution patterns can be read against model confidence.
        figsize: (width, height) in inches.
        dpi: Output resolution.
    """
    if channels is None:
        channels = all_wavelengths

    T = s_images.shape[0]
    n_ch = len(channels)
    t = np.arange(T)
    label_str = "flare" if flare_peak_time is not None else "non-flare"
    flare_idx = _flare_index(timestamps, flare_peak_time) if flare_peak_time is not None else None

    fig = plt.figure(figsize=figsize, dpi=dpi)
    gs = gridspec.GridSpec(
        n_ch + 1, 2,
        figure=fig,
        hspace=0.6,
        wspace=0.35,
        height_ratios=[1.4] + [1.0] * n_ch,
    )

    # ── GOES panel (spans both columns) ──────────────────────────────────────
    ax_goes = fig.add_subplot(gs[0, :])
    xlimits = (timestamps.min(), timestamps.max())
    plot_goes(goes_ts, columns=["xrsb"], xlimits=xlimits, ax=ax_goes)
    ax_goes.set_title(
        f"AARP {aarp_id}  ({label_str})  —  GOES 1–8 Å",
        fontsize=9,
    )
    if flare_peak_time is not None:
        ax_goes.axvline(
            pd.Timestamp(flare_peak_time),
            color="red", linestyle="--", linewidth=1, label="flare peak",
        )
    if pred_scores is not None:
        ax2 = ax_goes.twinx()
        ax2.plot(timestamps.values, pred_scores,
                 color="navy", linewidth=1, label="flare score")
        ax2.set_ylim(-0.05, 1.05)
        ax2.set_ylabel("flare score", fontsize=7, color="navy")
        ax2.tick_params(axis="y", labelsize=6, labelcolor="navy")
        ax2.legend(fontsize=7, loc="upper left")
    ax_goes.legend(fontsize=7)

    # ── Per-channel rows ──────────────────────────────────────────────────────
    for row, wavelength in enumerate(channels, start=1):
        ch_idx = all_wavelengths.index(wavelength)
        first_row = (row == 1)

        # ── Raw log-intensity: median ± IQR band + 99th pct ──────────────
        ax_raw = fig.add_subplot(gs[row, 0])
        raw_ch = np.log1p(np.maximum(s_images[:, ch_idx, :, :], 0))  # clip negatives: non-physical calibration artifacts
        q = _spatial_percentiles(raw_ch, RAW_PERCENTILES)

        ax_raw.fill_between(t, q[25], q[75],
                            alpha=0.25, color="steelblue", label="IQR (25–75th)")
        ax_raw.plot(t, q[50], color="steelblue", linewidth=1, label="median")
        ax_raw.plot(t, q[99], color="navy", linewidth=1,
                    linestyle="--", label="99th pct")
        if flare_idx is not None:
            ax_raw.axvline(flare_idx, color="red", linestyle="--",
                           linewidth=0.8, alpha=0.7)
        ax_raw.set_ylabel("log(DN + 1)", fontsize=7)
        ax_raw.set_title(f"{wavelength} Å  —  raw intensity", fontsize=8)
        ax_raw.tick_params(axis="y", labelsize=6)
        _set_time_xticks(ax_raw, timestamps)
        if first_row:
            ax_raw.legend(fontsize=6, loc="upper left")

        # ── Attribution upper tail: 95th and 99th pct ────────────────────
        ax_attr = fig.add_subplot(gs[row, 1])
        attr_ch = np.array([attributions[t_idx][ch_idx] for t_idx in range(T)])  # [T, H, W]
        qa = _spatial_percentiles(attr_ch, ATTR_PERCENTILES)

        ax_attr.axhline(0, color="black", linewidth=0.5, linestyle=":")
        ax_attr.plot(t, qa[95], color="coral", linewidth=1, label="95th pct")
        ax_attr.plot(t, qa[99], color="darkred", linewidth=1,
                     linestyle="--", label="99th pct")
        if flare_idx is not None:
            ax_attr.axvline(flare_idx, color="red", linestyle="--",
                            linewidth=0.8, alpha=0.7)
        ax_attr.set_ylabel("IG attribution", fontsize=7)
        ax_attr.set_title(f"{wavelength} Å  —  attribution (upper tail)", fontsize=8)
        ax_attr.tick_params(axis="y", labelsize=6)
        _set_time_xticks(ax_attr, timestamps)
        if first_row:
            ax_attr.legend(fontsize=6, loc="upper left")

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "raw_and_attribution_timeseries.png")
    fig.savefig(out_path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    print(f"Saved {out_path}")


def make_boxplot(aarp_id, metadata_df, transform, model, device, output_home=OUTPUT_HOME,
                 multiply_by_inputs=True, resize_to=None, stride=1):
    """Compute IG attributions and generate the quantile timeseries figure for one AARP.

    Args:
        aarp_id: Integer AARP identifier.
        metadata_df: DataFrame for the relevant split (test or validation).
        transform: AIALogTransform instance.
        model: Trained ViT model (must already be on the correct device).
        device: torch.device.
        output_home: Root directory; output goes to {output_home}/{aarp_id}/.
        multiply_by_inputs: Passed to IntegratedGradients. False gives pure gradient
            signal uncoupled from pixel brightness, useful for diagnosing whether
            attribution patterns are driven by input magnitude or model sensitivity.
        stride: Use every Nth frame (IG is the expensive step here, ~5-6s/frame on
            the pretrained architecture -- a 451-frame AARP takes ~40 minutes at
            stride=1). Applied consistently to images/attributions/flare-scores AND
            timestamps, so the x-axis stays aligned with the (thinned) data.
    """
    output_dir = os.path.join(output_home, str(aarp_id))

    s_images, attributions = run_pred_and_ig(
        aarp_id, metadata_df, transform=transform, model=model, device=device,
        multiply_by_inputs=multiply_by_inputs, resize_to=resize_to, stride=stride,
    )

    # Compute per-timestep flare probability — fast (no IG, forward passes only)
    tensor_images = torch.from_numpy(s_images).to(torch.float32)
    tensor_data = transform(tensor_images)
    if resize_to is not None:
        tensor_data = v2.Resize(resize_to)(tensor_data)
    tensor_data = tensor_data.to(device)
    dataset = TensorDataset(tensor_data)
    probs = make_predictions(dataset, model=model, device=device, probabilities=True)
    flare_scores = probs[:, 1].cpu().numpy()   # class 1 = flare, shape [T]
    del tensor_images, tensor_data, dataset

    torch.cuda.empty_cache()
    gc.collect()

    s_aarp = single_aarp(aarp_id, metadata_df.query(f"aarp_id == {aarp_id}"))
    timestamps = s_aarp.timestamps.reset_index(drop=True)
    if stride > 1:
        # Must match run_pred_and_ig's own s_images[::stride] slicing above, so the
        # x-axis stays aligned with the (thinned) images/attributions/flare-scores.
        timestamps = timestamps.iloc[::stride].reset_index(drop=True)

    fl_start, fl_end = get_start_and_end_time(
        s_aarp.get_midtime(), FLARE_TIME_WINDOW_HOURS * 60
    )
    goes_ts = aarp_ml.utils.run_fetch_goes(fl_start, fl_end)

    flare_peak_time = None
    goes_event_list_path = DatasetPaths(parent_dir=".").goes_event_with_aarp
    goes_event_list = pd.read_csv(
        goes_event_list_path,
        parse_dates=["event_date", "start_time", "peak_time", "end_time"],
    )
    matched = goes_event_list.query(f"harpnum == {aarp_id}")
    if not matched.empty:
        flare_peak_time = matched["peak_time"].iloc[0]

    plot_intensity_and_attribution_timeseries(
        aarp_id=aarp_id,
        s_images=s_images,
        attributions=attributions,
        timestamps=timestamps,
        goes_ts=goes_ts,
        flare_peak_time=flare_peak_time,
        pred_scores=flare_scores,
        output_dir=output_dir,
    )

    del attributions, s_images
    torch.cuda.empty_cache()
    gc.collect()


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Raw intensity + attribution quantile timeseries per AARP"
    )
    parser.add_argument("--json-path",  default="solar_dataset.json")
    parser.add_argument("--model-path", default=TRAINED_MODEL_PATH,
                        help="Path to trained ViT model checkpoint.")
    parser.add_argument("--output-dir", default=None,
                        help="Root directory for output plots. Defaults to "
                             "plots/intensity_boxplots/<run-id> derived from --model-path.")
    parser.add_argument("--aarp-id", type=int, nargs="+", default=None,
                        help="One or more AARP IDs to process (space-separated). "
                             "Omit to process the full test+val set.")
    parser.add_argument("--no-input-mult", action="store_true",
                        help="Use multiply_by_inputs=False in IG (pure gradient, no input weighting). "
                             "Output written to a separate directory for side-by-side comparison.")
    parser.add_argument("--model-type", default="vit", choices=list(VALID_MODEL_TYPES),
                        help="Model architecture: 'vit' (DeepFlare_ViT, default) or "
                             "'vit-pretrained' (torchvision vit_l_16).")
    parser.add_argument("--stride", type=int, default=1,
                        help="Use every Nth frame per AARP for IG (default: 1, all frames). "
                             "IG is the expensive step here (~5-6s/frame on the pretrained "
                             "architecture) -- e.g. a 451-frame AARP takes ~40 minutes at "
                             "stride=1. Try stride=4+ for a quick look before a full run.")
    args = parser.parse_args()

    from pathlib import Path
    multiply_by_inputs = not args.no_input_mult
    base_dir = args.output_dir
    if base_dir is None:
        run_id = Path(args.model_path).parent.name
        base_dir = f"plots/intensity_boxplots/{run_id}"
    output_dir = base_dir if multiply_by_inputs else base_dir.rstrip("/") + "_no_input_mult"

    config = TrainingConfig(json_path=args.json_path, stats_file="stats.pkl",
                            model_type=VALID_MODEL_TYPES[args.model_type])
    config.trained_model_path = args.model_path
    metadata, model, transform, device = get_data_model(config)
    resize_to = needs_resize(config)
    print(f"Model type : {args.model_type}" + (f"  (resize to {resize_to})" if resize_to else ""))
    _, val_df, test_df = dfs_from_metadata(metadata)

    os.makedirs(output_dir, exist_ok=True)

    if args.aarp_id is not None:
        for aarp_id in args.aarp_id:
            if not test_df.empty and aarp_id in test_df.aarp_id.values:
                make_boxplot(aarp_id, test_df, transform, model, device, output_dir,
                             multiply_by_inputs=multiply_by_inputs, resize_to=resize_to, stride=args.stride)
            elif not val_df.empty and aarp_id in val_df.aarp_id.values:
                make_boxplot(aarp_id, val_df, transform, model, device, output_dir,
                             multiply_by_inputs=multiply_by_inputs, resize_to=resize_to, stride=args.stride)
            else:
                print(f"AARP {aarp_id} not found in test or validation splits")
    else:
        for aarp_id in (test_df.aarp_id.unique().tolist() if not test_df.empty else []):
            make_boxplot(aarp_id, test_df, transform, model, device, output_dir,
                         multiply_by_inputs=multiply_by_inputs, resize_to=resize_to, stride=args.stride)
        for aarp_id in (val_df.aarp_id.unique().tolist() if not val_df.empty else []):
            make_boxplot(aarp_id, val_df, transform, model, device, output_dir,
                         multiply_by_inputs=multiply_by_inputs, resize_to=resize_to, stride=args.stride)


if __name__ == "__main__":
    main()
