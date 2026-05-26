"""Side-by-side ViT vs XGBoost comparison for paper reporting.

Produces publication-quality figures and a comparison table without
exposing internal run-id code names in any plot title or label.

Evaluation protocol
-------------------
  Image-level : ROC curve + AUC for both models overlaid on one axes.
                Labels are inherited from the AR (every frame of a flare
                AR is labelled 1).
  AR-level    : per-image probabilities aggregated to AR level by mean
                pooling (group by AR, average probabilities), then
                thresholded at 0.5.  Metrics: CM, precision, recall,
                specificity, F1, MCC, TSS, HSS.
  Baseline    : always-positive classifier included in the metrics table
                for context.

Outputs (all in --output-dir/)
-------------------------------
  roc_comparison.png       Overlaid ROC curves (image level)
  cm_comparison.png        Side-by-side AR-level CMs (mean aggregation)
  metrics_comparison.csv   Wide table: metric × model (ViT / XGBoost / baseline)

Usage
-----
    python -m src.torch.vit.compare_models \\
        --vit-path  outputs/glad-shape-197/trained_model.pth \\
        --xgb-path  outputs/devoted-pond-14/best_xgboost_model.json \\
        --subset validation test

    # Custom display labels
    python -m src.torch.vit.compare_models \\
        --vit-path  outputs/glad-shape-197/trained_model.pth \\
        --xgb-path  outputs/devoted-pond-14/best_xgboost_model.json \\
        --subset validation test \\
        --vit-label "ViT" --xgb-label "XGBoost"

    # VIT_Pretrained variant
    python -m src.torch.vit.compare_models \\
        --vit-path  outputs/treasured-blaze-221/trained_model.pth \\
        --vit-type  vit-pretrained \\
        --xgb-path  outputs/devoted-pond-14/best_xgboost_model.json \\
        --subset validation test
"""

import argparse
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from aarp_ml.torch.dataset import aia_euv
from ml_utils.visualization import plot_confusion_matrix
from src.torch.vit.evaluate_aarp import (
    CONFUSION_MATRIX_CLASSES,
    DEFAULT_THRESHOLD,
    N_CHANNELS,
    aggregate_per_aarp,
    compute_baseline_metrics,
    compute_skill_scores,
    load_vit,
    load_vit_pretrained,
    load_xgb,
    run_vit_inference,
    run_xgb_inference,
)
from src.torch.vit.utils import dfs_from_metadata, get_metadata_from_json

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_OUTPUT_DIR  = "plots/model_comparison"
DEFAULT_VIT_LABEL   = "ViT"
DEFAULT_XGB_LABEL   = "XGBoost"
DEFAULT_BATCH_SIZE  = 32
VALID_SUBSETS       = {"training", "validation", "test"}

# Consistent colors across all figures
_COLORS = {
    "vit":      "#1f77b4",   # blue
    "xgb":      "#d62728",   # red
    "baseline": "#7f7f7f",   # grey
    "random":   "#aaaaaa",   # light grey (diagonal)
}


# ---------------------------------------------------------------------------
# Inference helper (shared for ViT and XGB)
# ---------------------------------------------------------------------------

def _run_model(
    model_path:  str,
    model_type:  str,
    subsets:     List[str],
    json_path:   str,
    stats_file:  str,
    batch_size:  int,
    metadata:    dict,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load model, run inference, return (aarp_ids, y_true, y_prob)."""
    training_df, val_df, test_df = dfs_from_metadata(metadata)
    subset_df_map = {"training": training_df, "validation": val_df, "test": test_df}

    if model_type == "vit":
        model, transform, device = load_vit(model_path, stats_file)
        resize_to = None
    elif model_type == "vit-pretrained":
        model, transform, device = load_vit_pretrained(model_path, stats_file)
        resize_to = 224
    else:
        booster = load_xgb(model_path)

    all_ids, all_true, all_prob = [], [], []

    for subset in subsets:
        subset_df = subset_df_map[subset]

        if model_type in ("vit", "vit-pretrained"):
            dataset = aia_euv(json_path, subset=subset,
                              transform=v2.Compose([transform]))
            loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False)
            y_true, y_prob = run_vit_inference(model, loader, device,
                                               resize_to=resize_to)
        else:
            y_true, y_prob = run_xgb_inference(
                booster, json_path, stats_file, subset
            )

        if len(y_true) != len(subset_df):
            raise RuntimeError(
                f"Prediction count ({len(y_true)}) != DataFrame rows "
                f"({len(subset_df)}) for subset '{subset}'."
            )

        all_ids.extend(subset_df["aarp_id"].values)
        all_true.extend(y_true)
        all_prob.extend(y_prob)

    return (np.array(all_ids), np.array(all_true), np.array(all_prob))


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_roc_comparison(
    models: List[Dict],
    output_path: Path,
    dpi: int = 150,
) -> None:
    """Overlay ROC curves for all models on a single axes.

    Args:
        models: List of dicts, each with keys:
                  label     — display name (e.g. "ViT")
                  y_true    — per-frame ground-truth labels
                  y_prob    — per-frame P(flare) probabilities
                  color     — matplotlib color string
        output_path: Where to save the figure.
        dpi: Output resolution.
    """
    fig, ax = plt.subplots(figsize=(5, 5))

    for m in models:
        fpr, tpr, _ = roc_curve(m["y_true"], m["y_prob"])
        auc = roc_auc_score(m["y_true"], m["y_prob"])
        ax.plot(fpr, tpr, lw=1.8, color=m["color"],
                label=f"{m['label']}  (AUC = {auc:.3f})")

    ax.plot([0, 1], [0, 1], "--", lw=1, color=_COLORS["random"],
            label="Random classifier")

    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("ROC Curve (image level)", fontsize=12)
    ax.legend(loc="lower right", fontsize=10)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    print(f"  ✓ ROC comparison  → {output_path}")


def plot_cm_comparison(
    models: List[Dict],
    output_path: Path,
    dpi: int = 150,
) -> None:
    """Side-by-side AR-level confusion matrices (mean aggregation).

    Args:
        models: List of dicts with keys:
                  label — display name
                  agg   — per-AARP DataFrame with 'true_label', 'pred_mean'
        output_path: Where to save the figure.
        dpi: Output resolution.
    """
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, m in zip(axes, models):
        cm = confusion_matrix(
            m["agg"]["true_label"], m["agg"]["pred_mean"],
            labels=CONFUSION_MATRIX_CLASSES,
        )
        # reuse ml_utils helper but draw into our axes
        plot_confusion_matrix(
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
            figure=fig,
            axis=ax,
            fontcolor_threshold=0.5,
        )
        ax.set_title(m["label"], fontsize=12)

    fig.suptitle("AR-level Confusion Matrix (mean aggregation, threshold = 0.5)",
                 fontsize=11, y=1.02)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    print(f"  ✓ CM comparison   → {output_path}")


# ---------------------------------------------------------------------------
# Metrics table
# ---------------------------------------------------------------------------

def build_metrics_table(
    models:      List[Dict],
    agg_method:  str = "mean",
    subset_label: str = "",
) -> pd.DataFrame:
    """Build a wide metrics table: rows=metrics, columns=models + baseline.

    Args:
        models: List of dicts with keys label, agg, y_true_ar, y_prob_ar.
        agg_method: 'mean' or 'max' — which aggregated column to use.
        subset_label: Subset string for the index label.

    Returns:
        Wide DataFrame suitable for direct export to a paper table.
    """
    pred_col = f"pred_{agg_method}"
    prob_col = f"{agg_method}_prob"

    records: Dict[str, Dict] = {}

    for m in models:
        agg = m["agg"]
        cm  = confusion_matrix(
            agg["true_label"], agg[pred_col], labels=CONFUSION_MATRIX_CLASSES
        )
        scores = compute_skill_scores(
            cm, agg["true_label"].values,
            agg[prob_col].values, agg[pred_col].values
        )
        # Add image-level AUC
        scores["ROC-AUC"] = roc_auc_score(m["y_true"], m["y_prob"])
        records[m["label"]] = scores

    # Always-positive baseline (use first model's agg — same AR set)
    baseline = compute_baseline_metrics(models[0]["agg"])
    baseline["ROC-AUC"] = float("nan")
    records["Baseline\n(always positive)"] = baseline

    # Rows = metrics, columns = model labels
    metric_order = [
        "TP", "FP", "FN", "TN",
        "precision", "recall", "specificity", "F1", "MCC",
        "FAR", "TSS", "HSS", "accuracy", "ROC-AUC", "log_loss",
    ]
    df = pd.DataFrame(records).loc[metric_order]
    df.index.name = "metric"
    return df


def print_metrics_table(df: pd.DataFrame) -> None:
    sep = "=" * (16 + 12 * len(df.columns))
    print(f"\n{sep}")
    print("  AR-level metrics comparison")
    print(sep)
    float_fmt = lambda x: f"{x:.4f}" if isinstance(x, float) else str(x)
    print(df.to_string(float_format=lambda x: f"{x:.4f}"))
    print(sep)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare ViT and XGBoost — paper-quality figures and metrics.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--vit-path", required=True,
                        help="Path to ViT checkpoint (.pth).")
    parser.add_argument("--vit-type", default="vit",
                        choices=["vit", "vit-pretrained"],
                        help="ViT architecture variant (default: vit).")
    parser.add_argument("--vit-label", default=DEFAULT_VIT_LABEL,
                        help=f"Display name for ViT in plots (default: {DEFAULT_VIT_LABEL}).")
    parser.add_argument("--xgb-path", required=True,
                        help="Path to XGBoost model (.json).")
    parser.add_argument("--xgb-label", default=DEFAULT_XGB_LABEL,
                        help=f"Display name for XGBoost in plots (default: {DEFAULT_XGB_LABEL}).")
    parser.add_argument("--subset", nargs="+", required=True,
                        choices=list(VALID_SUBSETS),
                        help="Subset(s) to evaluate (e.g. --subset validation test).")
    parser.add_argument("--json-path",  default="solar_dataset.json")
    parser.add_argument("--stats-file", default="stats.pkl")
    parser.add_argument("--threshold",  type=float, default=DEFAULT_THRESHOLD,
                        help=f"AR-level decision threshold (default: {DEFAULT_THRESHOLD}).")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--output-dir", default=None,
                        help="Output directory. Defaults to "
                             "plots/model_comparison/<vit-run-id>_vs_<xgb-run-id>/<subset>/.")
    parser.add_argument("--exclude-aarp", nargs="+", type=int, default=None,
                        metavar="AARP_ID",
                        help="AARP IDs to exclude from AR-level evaluation.")
    parser.add_argument("--dpi", type=int, default=150,
                        help="Figure DPI (default: 150; use 300 for print).")
    args = parser.parse_args()

    subset_label = "+".join(args.subset)
    vit_run_id   = Path(args.vit_path).parent.name
    xgb_run_id   = Path(args.xgb_path).parent.name
    excl_suffix  = (
        "_excl" + "-".join(str(a) for a in sorted(args.exclude_aarp))
        if args.exclude_aarp else ""
    )

    if args.output_dir is None:
        args.output_dir = (
            f"{DEFAULT_OUTPUT_DIR}/{vit_run_id}_vs_{xgb_run_id}"
            f"/{subset_label}{excl_suffix}"
        )

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Subset(s)  : {subset_label}")
    print(f"Threshold  : {args.threshold}")
    print(f"Output dir : {out}")
    if args.exclude_aarp:
        print(f"Excluding  : {args.exclude_aarp}")

    # ── Load shared metadata ───────────────────────────────────────────────
    metadata = get_metadata_from_json(args.json_path)

    # ── Run inference for both models ──────────────────────────────────────
    print(f"\nRunning {args.vit_label} inference…")
    vit_ids, vit_true, vit_prob = _run_model(
        args.vit_path, args.vit_type, args.subset,
        args.json_path, args.stats_file, args.batch_size, metadata,
    )

    print(f"\nRunning {args.xgb_label} inference…")
    xgb_ids, xgb_true, xgb_prob = _run_model(
        args.xgb_path, "xgb", args.subset,
        args.json_path, args.stats_file, args.batch_size, metadata,
    )

    # ── Aggregate to AR level ──────────────────────────────────────────────
    vit_agg = aggregate_per_aarp(vit_ids, vit_true, vit_prob, args.threshold)
    xgb_agg = aggregate_per_aarp(xgb_ids, xgb_true, xgb_prob, args.threshold)

    if args.exclude_aarp:
        vit_agg = vit_agg[~vit_agg["aarp_id"].isin(args.exclude_aarp)].reset_index(drop=True)
        xgb_agg = xgb_agg[~xgb_agg["aarp_id"].isin(args.exclude_aarp)].reset_index(drop=True)
        # also filter frame-level arrays for ROC
        vit_mask = ~np.isin(vit_ids, args.exclude_aarp)
        xgb_mask = ~np.isin(xgb_ids, args.exclude_aarp)
        vit_true, vit_prob = vit_true[vit_mask], vit_prob[vit_mask]
        xgb_true, xgb_prob = xgb_true[xgb_mask], xgb_prob[xgb_mask]

    # ── Build model descriptors ────────────────────────────────────────────
    models = [
        {"label": args.vit_label, "color": _COLORS["vit"],
         "y_true": vit_true, "y_prob": vit_prob, "agg": vit_agg},
        {"label": args.xgb_label, "color": _COLORS["xgb"],
         "y_true": xgb_true, "y_prob": xgb_prob, "agg": xgb_agg},
    ]

    # ── Figures ────────────────────────────────────────────────────────────
    print("\nGenerating figures…")
    plot_roc_comparison(models, out / "roc_comparison.png", dpi=args.dpi)
    plot_cm_comparison(models,  out / "cm_comparison.png",  dpi=args.dpi)

    # ── Metrics table ──────────────────────────────────────────────────────
    df = build_metrics_table(models, agg_method="mean",
                             subset_label=subset_label)
    print_metrics_table(df)

    csv_path = out / "metrics_comparison.csv"
    df.to_csv(csv_path)
    print(f"\n  ✓ Metrics table   → {csv_path}")


if __name__ == "__main__":
    main()
