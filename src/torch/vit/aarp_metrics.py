"""Per-AARP aggregated metrics and confusion matrix.

Runs frame-level inference on a subset, then aggregates predictions per AARP
via majority vote and mean flare probability. Produces:
  - Printed per-AARP metrics table
  - CSV: plots/aarp_metrics_<subset>.csv
  - AARP-level CM plot: plots/cm_aarp_<subset>.png

Usage:
    python -m src.torch.vit.aarp_metrics --subset test
    python -m src.torch.vit.aarp_metrics --subset validation
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from tqdm import tqdm

from aarp_ml.torch.dataset import aia_euv
from ml_utils.visualization import plot_confusion_matrix
from src.torch.vit.test import CONFUSION_MATRIX_CLASSES, compute_metrics
from src.torch.vit.train import TrainingConfig
from src.torch.vit.utils import dfs_from_metadata, get_data_model

import matplotlib.pyplot as plt

DEFAULT_JSON_PATH = "solar_dataset.json"
DEFAULT_STATS_FILE = "stats.pkl"
DEFAULT_TRAINED_MODEL_PATH = "outputs/glad-shape-197/trained_model.pth"
DEFAULT_BATCH_SIZE = 32
DEFAULT_OUTPUT_DIR = "plots"
VALID_SUBSETS = {"training", "validation", "test"}


def run_inference(model, data_loader, device):
    """Return parallel lists of true labels, predicted labels, and flare probabilities."""
    model.eval()
    y_true, y_pred, y_prob = [], [], []

    with torch.inference_mode():
        for images, labels in tqdm(data_loader, desc="Running inference", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)[:, 1]  # P(flare)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            y_prob.extend(probs.cpu().numpy())

    return np.array(y_true), np.array(y_pred), np.array(y_prob)


def build_aarp_table(subset_df, y_true, y_pred, y_prob):
    """Attach predictions to AARP IDs and aggregate per AARP."""
    df = subset_df[["aarp_id", "label"]].copy().reset_index(drop=True)
    df["y_pred"] = y_pred
    df["y_prob"] = y_prob

    agg = (
        df.groupby("aarp_id")
        .agg(
            true_label=("label", "first"),
            n_frames=("label", "count"),
            n_pred_flare=("y_pred", "sum"),
            mean_prob=("y_prob", "mean"),
        )
        .reset_index()
    )
    agg["majority_pred"] = (agg["n_pred_flare"] > agg["n_frames"] / 2).astype(int)
    agg["correct"] = agg["true_label"] == agg["majority_pred"]
    agg = agg.sort_values(["true_label", "aarp_id"]).reset_index(drop=True)
    return agg


def print_aarp_table(agg):
    pd.set_option("display.max_rows", None)
    pd.set_option("display.float_format", "{:.3f}".format)
    print("\nPer-AARP Metrics")
    print("=" * 70)
    print(
        agg[["aarp_id", "true_label", "n_frames", "n_pred_flare", "majority_pred", "mean_prob", "correct"]]
        .to_string(index=False)
    )
    print()


def save_aarp_cm(agg, subset, output_dir):
    cm = confusion_matrix(agg["true_label"], agg["majority_pred"], labels=CONFUSION_MATRIX_CLASSES)
    metrics = compute_metrics(cm)

    print(f"AARP-level Confusion Matrix ({subset})")
    print("=" * 50)
    print(cm)
    print(f"\n  TP: {metrics['TP']}  FP: {metrics['FP']}  FN: {metrics['FN']}  TN: {metrics['TN']}")
    print(f"  Precision: {metrics['precision']:.4f}  Recall: {metrics['recall']:.4f}")
    total = len(agg)
    accuracy = (metrics["TP"] + metrics["TN"]) / total
    print(f"  Accuracy: {accuracy:.4f}  (over {total} AARPs)")
    print("=" * 50)

    fig, _ = plot_confusion_matrix(
        cm,
        hide_spines=False,
        hide_ticks=False,
        figsize=None,
        cmap=None,
        colorbar=False,
        show_absolute=True,
        show_normed=False,
        norm_colormap=None,
        class_names=None,
        figure=None,
        axis=None,
        fontcolor_threshold=0.5,
    )

    out_path = Path(output_dir) / f"cm_aarp_{subset}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"\n✓ Saved AARP-level CM to {out_path}")


def main(subset, json_path, stats_file, model_path, batch_size, output_dir):
    config = TrainingConfig(
        json_path=json_path,
        stats_file=stats_file,
        trained_model_path=model_path,
    )

    print("Loading model and metadata...")
    metadata, model, transform, device = get_data_model(config)
    model.to(device)

    training_df, val_df, test_df = dfs_from_metadata(metadata)
    subset_df_map = {"training": training_df, "validation": val_df, "test": test_df}
    subset_df = subset_df_map[subset]

    dataset = aia_euv(json_path, subset=subset, transform=v2.Compose([transform]))
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    y_true, y_pred, y_prob = run_inference(model, data_loader, device)

    if len(y_true) != len(subset_df):
        raise RuntimeError(
            f"Prediction count ({len(y_true)}) does not match DataFrame rows ({len(subset_df)}). "
            "Ensure shuffle=False and JSON order is preserved."
        )

    agg = build_aarp_table(subset_df, y_true, y_pred, y_prob)

    print_aarp_table(agg)

    csv_path = Path(output_dir) / f"aarp_metrics_{subset}.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(csv_path, index=False)
    print(f"✓ Saved per-AARP metrics to {csv_path}")

    save_aarp_cm(agg, subset, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Per-AARP aggregated metrics and confusion matrix.")
    parser.add_argument("--subset", required=True, choices=list(VALID_SUBSETS))
    parser.add_argument("--json-path", default=DEFAULT_JSON_PATH)
    parser.add_argument("--stats-file", default=DEFAULT_STATS_FILE)
    parser.add_argument("--model-path", default=DEFAULT_TRAINED_MODEL_PATH)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    main(
        subset=args.subset,
        json_path=args.json_path,
        stats_file=args.stats_file,
        model_path=args.model_path,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
    )
