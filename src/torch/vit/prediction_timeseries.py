"""GOES X-ray flux + model flare score timeseries per AARP.

Lightweight alternative to intensity_boxplot.py — no IG computation, just
forward passes. Runs in minutes rather than hours per AARP.

Always plots probs[:,1] (flare class probability) for both flare and non-flare
AARPs so all events sit on the same 0–1 scale and are directly comparable:
  - flare AARPs:     score rises toward 1.0 near the flare peak
  - non-flare AARPs: score stays near 0.0 throughout

Outputs:
    plots/prediction_timeseries/{aarp_id}/prediction_timeseries.png
"""

import os
import gc
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import aarp_ml

from torch.utils.data import TensorDataset
from astro_utils.utils import get_start_and_end_time
from src.torch.vit.ig import single_aarp, make_predictions
from src.torch.vit.predictions_analyze import plot_goes, FLARE_TIME_WINDOW_HOURS
from src.torch.vit.utils import get_data_model, dfs_from_metadata
from src.torch.vit.train import TrainingConfig
from src.data_single import DatasetPaths

TRAINED_MODEL_PATH: str = "outputs/glad-shape-197/trained_model.pth"
OUTPUT_HOME: str = "plots/prediction_timeseries"
FIGSIZE: tuple = (12, 4)
OUTPUT_DPI: int = 150


def make_prediction_timeseries(aarp_id, metadata_df, transform, model, device,
                                output_home=OUTPUT_HOME):
    """Generate GOES + flare score plot for one AARP without running IG.

    Args:
        aarp_id: Integer AARP identifier.
        metadata_df: DataFrame for the relevant split (test or validation).
        transform: AIALogTransform instance.
        model: Trained ViT model (must already be on the correct device).
        device: torch.device.
        output_home: Root directory; output goes to {output_home}/{aarp_id}/.
    """
    output_dir = os.path.join(output_home, str(aarp_id))
    os.makedirs(output_dir, exist_ok=True)

    # Load images — no IG
    s_aarp = single_aarp(aarp_id, metadata_df.query(f"aarp_id == {aarp_id}"))
    s_images = s_aarp.get_images()
    timestamps = s_aarp.timestamps.reset_index(drop=True)
    label = s_aarp.label
    label_str = "flare" if label == 1 else "non-flare"

    # Forward passes only — fast
    tensor_images = torch.from_numpy(s_images).to(torch.float32)
    tensor_data = transform(tensor_images).to(device)
    dataset = TensorDataset(tensor_data)
    probs = make_predictions(dataset, model=model, device=device, probabilities=True)
    flare_scores = probs[:, 1].cpu().numpy()   # always flare-class probability [T]
    del tensor_images, tensor_data, dataset, s_images
    torch.cuda.empty_cache()
    gc.collect()

    # GOES timeseries
    fl_start, fl_end = get_start_and_end_time(
        s_aarp.get_midtime(), FLARE_TIME_WINDOW_HOURS * 60
    )
    goes_ts = aarp_ml.utils.run_fetch_goes(fl_start, fl_end)

    # Flare peak time if positive class
    flare_peak_time = None
    goes_event_list_path = DatasetPaths(parent_dir=".").goes_event_with_aarp
    goes_event_list = pd.read_csv(
        goes_event_list_path,
        parse_dates=["event_date", "start_time", "peak_time", "end_time"],
    )
    matched = goes_event_list.query(f"harpnum == {aarp_id}")
    if not matched.empty:
        flare_peak_time = matched["peak_time"].iloc[0]

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=OUTPUT_DPI)
    xlimits = (timestamps.min(), timestamps.max())
    plot_goes(goes_ts, columns=["xrsb"], xlimits=xlimits, ax=ax)

    if flare_peak_time is not None:
        ax.axvline(pd.Timestamp(flare_peak_time),
                   color="red", linestyle="--", linewidth=1, label="flare peak")

    ax2 = ax.twinx()
    ax2.plot(timestamps.values, flare_scores,
             color="navy", linewidth=1, label="flare score")
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_ylabel("flare score", fontsize=8, color="navy")
    ax2.tick_params(axis="y", labelsize=7, labelcolor="navy")
    ax2.legend(fontsize=7, loc="upper left")

    ax.set_title(
        f"AARP {aarp_id}  ({label_str})  —  GOES 1–8 Å + flare score",
        fontsize=9,
    )
    ax.legend(fontsize=7)

    out_path = os.path.join(output_dir, "prediction_timeseries.png")
    fig.savefig(out_path, bbox_inches="tight", dpi=OUTPUT_DPI)
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="GOES + flare score timeseries (no IG)"
    )
    parser.add_argument("--json-path", default="solar_dataset.json")
    parser.add_argument("--output-dir", default=OUTPUT_HOME)
    parser.add_argument("--aarp-id", type=int, nargs="+", default=None,
                        help="One or more AARP IDs (space-separated). "
                             "Omit to process the full test+val set.")
    args = parser.parse_args()

    config = TrainingConfig(json_path=args.json_path, stats_file="stats.pkl")
    config.trained_model_path = TRAINED_MODEL_PATH
    metadata, model, transform, device = get_data_model(config)
    model = model.to(device)
    _, val_df, test_df = dfs_from_metadata(metadata)

    os.makedirs(args.output_dir, exist_ok=True)

    def _run(aarp_id):
        if not test_df.empty and aarp_id in test_df.aarp_id.values:
            make_prediction_timeseries(aarp_id, test_df, transform, model, device,
                                       args.output_dir)
        elif not val_df.empty and aarp_id in val_df.aarp_id.values:
            make_prediction_timeseries(aarp_id, val_df, transform, model, device,
                                       args.output_dir)
        else:
            print(f"AARP {aarp_id} not found in test or validation splits")

    if args.aarp_id is not None:
        for aarp_id in args.aarp_id:
            _run(aarp_id)
    else:
        for aarp_id in (test_df.aarp_id.unique().tolist() if not test_df.empty else []):
            _run(aarp_id)
        for aarp_id in (val_df.aarp_id.unique().tolist() if not val_df.empty else []):
            _run(aarp_id)


if __name__ == "__main__":
    main()
