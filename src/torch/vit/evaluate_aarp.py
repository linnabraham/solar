"""AARP-level evaluation for ViT and XGBoost models.

Motivation
----------
Frame-level confusion matrices are biased: a long AARP sequence contributes
many more votes than a short one, and temporally correlated frames from the
same region are not independent samples.  The scientifically correct unit is
the AARP itself — the question is "will *this* active region produce a flare?",
not "does *this frame* look like a flare?"

This script aggregates per-frame model scores to the AARP level using two
strategies and produces a CM and standard skill scores for each:

  max   — max P(flare) over all frames.  "Did the model ever detect danger?"
           Conservative on FN; good for early-warning use.
  mean  — mean P(flare) over all frames.  Overall region-level confidence.
           More robust to single-frame spikes.

Both strategies are applied so you can compare them.  Use the same
aggregation method when comparing ViT vs XGBoost so the numbers are
apples-to-apples.

Supported models
----------------
  --model-type vit           DeepFlare_ViT (vit-pytorch), default
  --model-type vit-pretrained  torchvision vit_l_16 fine-tuned (VIT_Pretrained)
  --model-type xgb           XGBoost

Outputs (all under --output-dir/<run-id>/<subset-label>/)
----------------------------------------------------------
  image_level_curves.png  ROC + PR curves (frame level, labels inherited from AR)
  cm_max.png              AR-level CM using max aggregation
  cm_mean.png             AR-level CM using mean aggregation
  aarp_metrics.csv        Per-AARP table: true label, n_frames, max/mean prob, predictions
  skill_scores.csv        Model (max+mean) + always-positive baseline:
                          precision, recall, specificity, F1, MCC, TSS, HSS,
                          ROC-AUC, average precision, log-loss

Usage
-----
    # ViT, test set
    python -m src.torch.vit.evaluate_aarp \\
        --model-path outputs/glad-shape-197/trained_model.pth \\
        --subset test

    # ViT, combined val+test
    python -m src.torch.vit.evaluate_aarp \\
        --model-path outputs/glad-shape-197/trained_model.pth \\
        --subset validation test

    # VIT_Pretrained
    python -m src.torch.vit.evaluate_aarp \\
        --model-path outputs/treasured-blaze-221/trained_model.pth \\
        --model-type vit-pretrained \\
        --subset test

    # XGBoost
    python -m src.torch.vit.evaluate_aarp \\
        --model-path outputs/devoted-pond-14/best_xgboost_model.json \\
        --model-type xgb \\
        --subset test
"""

import argparse
import json
import pickle
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    confusion_matrix,
    matthews_corrcoef,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    log_loss,
)
from torch.utils.data import ConcatDataset, DataLoader
from torchvision.transforms import v2
from tqdm import tqdm

from aarp_ml.torch.dataset import AIALogTransform, aia_euv
from aarp_ml.torch.model import build_pretrained_vit
from ml_utils.visualization import plot_confusion_matrix
from src.torch.vit.utils import dfs_from_metadata, get_metadata_from_json

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_CHANNELS = 7
N_CLASSES = 2
CONFUSION_MATRIX_CLASSES = [0, 1]
DEFAULT_BATCH_SIZE = 32
DEFAULT_THRESHOLD = 0.5
DEFAULT_OUTPUT_DIR = "plots/aarp_eval"
VALID_SUBSETS = {"training", "validation", "test"}
VALID_MODEL_TYPES = {"vit", "vit-pretrained", "xgb"}


# ---------------------------------------------------------------------------
# Skill scores
# ---------------------------------------------------------------------------

def compute_skill_scores(cm: np.ndarray,
                         y_true: np.ndarray,
                         y_prob: np.ndarray,
                         y_pred: np.ndarray) -> Dict[str, float]:
    """Compute a full set of binary-classification skill scores from a 2×2 CM.

    Uses sklearn.metrics throughout.  Includes TSS and HSS (standard in
    solar-flare prediction literature), MCC, and specificity.

    Args:
        cm:     2×2 confusion matrix [[TN, FP], [FN, TP]].
        y_true: Ground-truth binary labels (AARP level).
        y_prob: Continuous P(flare) probabilities (for log-loss).
        y_pred: Thresholded binary predictions (for MCC).

    Returns:
        Dict with keys: TP, FP, FN, TN, precision, recall, specificity,
        FAR, F1, MCC, accuracy, TSS, HSS, log_loss.
        Undefined ratios (0/0) return 0.0.
    """
    TN, FP, FN, TP = cm.ravel()

    precision   = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall      = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0   # true-negative rate
    far         = 1.0 - specificity                             # FP / (FP + TN)
    f1          = (2 * precision * recall / (precision + recall)
                   if (precision + recall) > 0 else 0.0)
    accuracy    = (TP + TN) / len(y_true) if len(y_true) > 0 else 0.0
    balanced_accuracy = (recall + specificity) / 2.0
    # Base rate -- always read precision against this. At low prevalence, even a
    # precision well below 0.5 can be many multiples better than a random guess.
    prevalence  = (TP + FN) / len(y_true) if len(y_true) > 0 else 0.0
    tss         = recall - far

    # HSS = 2(TP·TN − FP·FN) / ((TP+FN)(FN+TN) + (TP+FP)(FP+TN))
    hss_denom = (TP + FN) * (FN + TN) + (TP + FP) * (FP + TN)
    hss       = 2 * (TP * TN - FP * FN) / hss_denom if hss_denom > 0 else 0.0

    # MCC via sklearn (handles edge cases cleanly)
    mcc = matthews_corrcoef(y_true, y_pred)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            bce = log_loss(y_true, y_prob)
        except Exception:
            bce = float("nan")

    return {
        "TP": int(TP), "FP": int(FP), "FN": int(FN), "TN": int(TN),
        "precision":   precision,
        "recall":      recall,
        "specificity": specificity,
        "FAR":         far,
        "F1":          f1,
        "MCC":         float(mcc),
        "accuracy":    accuracy,
        "balanced_accuracy": balanced_accuracy,
        "prevalence":  prevalence,
        "TSS":         tss,
        "HSS":         hss,
        "log_loss":    bce,
    }


# ---------------------------------------------------------------------------
# AARP-level aggregation
# ---------------------------------------------------------------------------

def aggregate_per_aarp(
    aarp_ids: np.ndarray,
    y_true:   np.ndarray,
    y_prob:   np.ndarray,
    threshold: float = DEFAULT_THRESHOLD,
) -> pd.DataFrame:
    """Collapse frame-level predictions to one row per AARP.

    Args:
        aarp_ids:  Per-frame AARP identifiers (same length as y_true / y_prob).
        y_true:    Per-frame true labels (0/1).
        y_prob:    Per-frame P(flare) scores in [0, 1].
        threshold: Decision boundary applied to aggregated scores.

    Returns:
        DataFrame with columns:
            aarp_id, true_label, n_frames,
            max_prob, mean_prob,
            pred_max, pred_mean
    """
    df = pd.DataFrame({
        "aarp_id": aarp_ids,
        "true_label": y_true,
        "y_prob": y_prob,
    })

    agg = (
        df.groupby("aarp_id")
        .agg(
            true_label=("true_label", "first"),
            n_frames=("true_label", "count"),
            max_prob=("y_prob", "max"),
            mean_prob=("y_prob", "mean"),
        )
        .reset_index()
    )

    agg["pred_max"]  = (agg["max_prob"]  > threshold).astype(int)
    agg["pred_mean"] = (agg["mean_prob"] > threshold).astype(int)
    agg = agg.sort_values(["true_label", "aarp_id"]).reset_index(drop=True)
    return agg


# ---------------------------------------------------------------------------
# ViT inference
# ---------------------------------------------------------------------------

def _load_transform(stats_file: str) -> AIALogTransform:
    with open(stats_file, "rb") as f:
        stats = pickle.load(f)
    means = [stats["mean"][f"channel_{i}"] for i in range(N_CHANNELS)]
    stds  = [stats["std"][f"channel_{i}"]  for i in range(N_CHANNELS)]
    return AIALogTransform(means=means, stds=stds)


def load_vit(model_path: str,
             stats_file: str) -> Tuple[torch.nn.Module, AIALogTransform, torch.device]:
    """Load a DeepFlare_ViT checkpoint."""
    from aarp_ml.torch.model import DeepFlare_ViT

    transform = _load_transform(stats_file)
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DeepFlare_ViT(
        height=512,            # fixed AIA image height
        n_classes=N_CLASSES,
        n_passbands=N_CHANNELS,
    ).model

    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)

    model.to(device).eval()
    return model, transform, device


def load_vit_pretrained(model_path: str,
                        stats_file: str) -> Tuple[torch.nn.Module, AIALogTransform, torch.device]:
    """Load a VIT_Pretrained (torchvision vit_l_16) checkpoint."""
    transform = _load_transform(stats_file)
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_pretrained_vit(n_channels=N_CHANNELS, n_classes=N_CLASSES, pretrained=False)

    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model, transform, device


def run_vit_inference(
    model:       torch.nn.Module,
    data_loader: DataLoader,
    device:      torch.device,
    resize_to:   Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run forward pass and return (y_true, y_prob) arrays.

    Args:
        model:       Model in eval mode, already on device.
        data_loader: DataLoader yielding (images, labels) with shuffle=False.
        device:      Torch device.
        resize_to:   If given (e.g. 224 for VIT_Pretrained), resize images
                     before inference.  Skipped when None.

    Returns:
        y_true: Per-frame ground-truth labels.
        y_prob: Per-frame P(flare) probabilities.
    """
    resize = v2.Resize(resize_to) if resize_to is not None else None

    y_true, y_prob = [], []
    with torch.inference_mode():
        for images, labels in tqdm(data_loader, desc="Inference", leave=False):
            images = images.to(device)
            if resize is not None:
                images = resize(images)
            outputs = model(images)
            probs   = F.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            y_true.extend(labels.numpy())
            y_prob.extend(probs)

    return np.array(y_true), np.array(y_prob)


# ---------------------------------------------------------------------------
# XGBoost inference
# ---------------------------------------------------------------------------

def load_xgb(model_path: str):
    """Load a saved XGBoost booster."""
    import xgboost as xgb
    booster = xgb.Booster()
    booster.load_model(model_path)
    return booster


def run_xgb_inference(
    booster,
    json_path:  str,
    stats_file: str,
    subset:     str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract per-frame features and return (y_true, y_prob).

    Args:
        booster:    Loaded XGBoost booster.
        json_path:  Dataset JSON path.
        stats_file: Normalisation stats pickle.
        subset:     Split name ('training', 'validation', or 'test').

    Returns:
        y_true: Per-frame ground-truth labels.
        y_prob: Per-frame P(flare) probabilities from XGBoost.
    """
    import xgboost as xgb
    from src.torch.xgb_train import (TrainingConfig as XGBConfig,
                                     extract_stats_generator,
                                     get_feature_names)

    transform = _load_transform(stats_file)
    dataset   = aia_euv(json_path, subset=subset, transform=transform)
    loader    = DataLoader(dataset, batch_size=1, shuffle=False)

    xgb_cfg      = XGBConfig(json_path=json_path, stats_file=stats_file)
    feature_names = get_feature_names(xgb_cfg)

    print(f"  Extracting XGBoost features for '{subset}' ({len(dataset)} frames)…")
    X, y = extract_stats_generator(loader, xgb_cfg)

    dmat  = xgb.DMatrix(X, label=y, feature_names=feature_names)
    probs = booster.predict(dmat)   # P(flare) directly from XGBoost
    return y, probs


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Image-level ROC + PR curves
# ---------------------------------------------------------------------------

def plot_image_level_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    output_dir: Path,
    label: str = "",
) -> Dict[str, float]:
    """Compute and save ROC and Precision-Recall curves at the image level.

    Labels are inherited from the AR: every frame of a flare AR gets label=1,
    every frame of a non-flare AR gets label=0.  This makes the curves
    informative about frame-level discriminability but does NOT reflect
    AR-level performance (use AR-level CM for that).

    Args:
        y_true:     Per-frame ground-truth labels.
        y_prob:     Per-frame P(flare) probabilities.
        output_dir: Directory to write PNG files.
        label:      Subtitle string for the plots.

    Returns:
        Dict with roc_auc and average_precision.
    """
    roc_auc = roc_auc_score(y_true, y_prob)
    ap      = average_precision_score(y_true, y_prob)

    fpr, tpr, _         = roc_curve(y_true, y_prob)
    prec, rec, _        = precision_recall_curve(y_true, y_prob)
    baseline_prevalence = y_true.mean()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle(f"Image-level curves  {label}", fontsize=10)

    # ROC
    ax = axes[0]
    ax.plot(fpr, tpr, lw=1.5, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC curve")
    ax.legend(loc="lower right")
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1])

    # PR
    ax = axes[1]
    ax.plot(rec, prec, lw=1.5, label=f"AP = {ap:.3f}")
    ax.axhline(baseline_prevalence, color="k", ls="--", lw=0.8,
               label=f"Baseline (prevalence = {baseline_prevalence:.2f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall curve")
    ax.legend(loc="upper right")
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1])

    plt.tight_layout()
    out_path = Path(output_dir) / "image_level_curves.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  ✓ Image-level curves → {out_path}")
    print(f"    ROC-AUC = {roc_auc:.4f}   Average Precision = {ap:.4f}")

    return {"roc_auc": roc_auc, "average_precision": ap}


# ---------------------------------------------------------------------------
# Always-positive baseline
# ---------------------------------------------------------------------------

def compute_baseline_metrics(agg: pd.DataFrame) -> Dict[str, float]:
    """Metrics for an always-positive (majority-class) baseline classifier.

    Predicts flare=1 for every AR regardless of input.  Contextualises model
    performance: any metric the model cannot beat over this baseline is not
    useful.

    Args:
        agg: Per-AARP aggregation DataFrame with 'true_label' column.

    Returns:
        Same keys as compute_skill_scores, prefixed context for printing.
    """
    n   = len(agg)
    y_t = agg["true_label"].values
    y_p = np.ones(n, dtype=int)          # always predict positive
    y_s = np.ones(n, dtype=float)        # probability = 1.0 always

    cm  = confusion_matrix(y_t, y_p, labels=CONFUSION_MATRIX_CLASSES)
    return compute_skill_scores(cm, y_t, y_s, y_p)


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _print_scores(label: str, agg_method: str, scores: Dict,
                  roc_auc: float = None, avg_precision: float = None) -> None:
    sep = "=" * 58
    print(f"\n{sep}")
    print(f"  {label}  [{agg_method} aggregation]")
    print(sep)
    print(f"  TP: {scores['TP']:4d}   FP: {scores['FP']:4d}")
    print(f"  FN: {scores['FN']:4d}   TN: {scores['TN']:4d}")
    print(f"  Prevalence  : {scores['prevalence']:.4f}  (base rate -- read Precision against this)")
    print(f"  Precision   : {scores['precision']:.4f}")
    print(f"  Recall      : {scores['recall']:.4f}")
    print(f"  Specificity : {scores['specificity']:.4f}")
    print(f"  FAR         : {scores['FAR']:.4f}")
    print(f"  F1          : {scores['F1']:.4f}")
    print(f"  MCC         : {scores['MCC']:.4f}")
    print(f"  Accuracy    : {scores['accuracy']:.4f}")
    print(f"  Balanced Acc: {scores['balanced_accuracy']:.4f}  ((Recall + Specificity) / 2)")
    print(f"  TSS         : {scores['TSS']:.4f}  (Recall − FAR)")
    print(f"  HSS         : {scores['HSS']:.4f}  (Heidke Skill Score)")
    print(f"  Log-loss    : {scores['log_loss']:.4f}")
    if roc_auc is not None:
        print(f"  ROC-AUC     : {roc_auc:.4f}")
    if avg_precision is not None:
        print(f"  Avg Precision: {avg_precision:.4f}")
    print(sep)


def _save_cm(cm: np.ndarray, title: str, out_path: Path) -> None:
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
    fig.suptitle(title, fontsize=10, y=1.02)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  ✓ CM saved → {out_path}")


def save_all_outputs(
    agg:          pd.DataFrame,
    subset_label: str,
    model_label:  str,
    output_dir:   str,
    threshold:    float,
    frame_y_true: np.ndarray,
    frame_y_prob: np.ndarray,
) -> None:
    """Save all evaluation outputs.

    Produces:
      image_level_curves.png  — ROC and PR curves at the frame level
      cm_image_level.png      — raw frame-level CM at `threshold`, no AARP aggregation
      cm_max.png / cm_mean.png — AR-level CMs for each aggregation method
      aarp_metrics.csv         — per-AARP table
      skill_scores.csv         — model + baseline metrics side-by-side (incl. image-level row)
      frame_predictions.npz    — raw (y_true, y_prob) per frame, so precision/recall at any
                                  other threshold can be recomputed later without rerunning
                                  the model
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1. Image-level ROC + PR curves
    print("\n── Image-level curves ──────────────────────────────────────")
    img_scores = plot_image_level_curves(
        frame_y_true, frame_y_prob, out,
        label=f"({model_label} | {subset_label})"
    )

    # 1b. Persist raw per-frame predictions -- lets precision/recall/CM be recomputed at any
    # threshold later without a fresh model pass.
    npz_path = out / "frame_predictions.npz"
    np.savez(npz_path, y_true=frame_y_true, y_prob=frame_y_prob, threshold=threshold)
    print(f"  ✓ Raw frame predictions → {npz_path}")

    # 1c. Raw frame-level CM at `threshold`, no AARP aggregation at all -- the rawest possible
    # view, distinct from both the max- and mean-aggregation AR-level views below.
    frame_y_pred = (frame_y_prob >= threshold).astype(int)
    frame_cm = confusion_matrix(frame_y_true, frame_y_pred, labels=CONFUSION_MATRIX_CLASSES)
    frame_scores = compute_skill_scores(frame_cm, frame_y_true, frame_y_prob, frame_y_pred)
    _print_scores(f"{model_label} | {subset_label}", "image-level (raw, no AARP pooling)",
                  frame_scores, roc_auc=img_scores["roc_auc"],
                  avg_precision=img_scores["average_precision"])
    _save_cm(
        frame_cm,
        title=f"{model_label} — {subset_label} (image-level, threshold={threshold})",
        out_path=out / "cm_image_level.png",
    )

    # 2. Per-AARP CSV
    csv_path = out / "aarp_metrics.csv"
    agg.to_csv(csv_path, index=False)
    print(f"  ✓ Per-AARP table → {csv_path}")

    # 3. AR-level CMs + skill scores (model)
    skill_rows = [{
        "classifier": model_label,
        "subset":     subset_label,
        "aggregation": "image_level",
        "n_samples":  len(frame_y_true),
        "roc_auc":    img_scores["roc_auc"],
        "avg_precision": img_scores["average_precision"],
        **frame_scores,
    }]
    for method, pred_col, prob_col in [
        ("max",  "pred_max",  "max_prob"),
        ("mean", "pred_mean", "mean_prob"),
    ]:
        cm = confusion_matrix(
            agg["true_label"], agg[pred_col], labels=CONFUSION_MATRIX_CLASSES
        )
        scores = compute_skill_scores(
            cm,
            agg["true_label"].values,
            agg[prob_col].values,
            agg[pred_col].values,
        )
        _print_scores(f"{model_label} | {subset_label}", method, scores,
                      roc_auc=img_scores["roc_auc"], avg_precision=img_scores["average_precision"])

        _save_cm(
            cm,
            title=f"{model_label} — {subset_label} ({method} aggregation, threshold={threshold})",
            out_path=out / f"cm_{method}.png",
        )

        skill_rows.append({
            "classifier": model_label,
            "subset":     subset_label,
            "aggregation": method,
            "n_samples":  len(agg),
            "roc_auc":    img_scores["roc_auc"],
            "avg_precision": img_scores["average_precision"],
            **scores,
        })

    # 4. Always-positive baseline
    baseline = compute_baseline_metrics(agg)
    _print_scores(f"BASELINE (always-positive) | {subset_label}", "—", baseline)
    skill_rows.append({
        "classifier": "always_positive",
        "subset":     subset_label,
        "aggregation": "—",
        "n_samples":  len(agg),
        "roc_auc":    float("nan"),
        "avg_precision": float("nan"),
        **baseline,
    })

    skill_path = out / "skill_scores.csv"
    pd.DataFrame(skill_rows).to_csv(skill_path, index=False)
    print(f"\n  ✓ Skill scores  → {skill_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="AARP-level evaluation for ViT and XGBoost models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model-path", required=True,
        help="Path to trained model checkpoint (.pth for ViT, .json for XGBoost).",
    )
    parser.add_argument(
        "--model-type", default="vit",
        choices=list(VALID_MODEL_TYPES),
        help="Model architecture (default: vit).",
    )
    parser.add_argument(
        "--subset", nargs="+", required=True,
        choices=list(VALID_SUBSETS),
        help="Subset(s) to evaluate. Pass multiple to combine "
             "(e.g. --subset validation test).",
    )
    parser.add_argument("--json-path",   default="solar_dataset.json")
    parser.add_argument("--stats-file",  default="stats.pkl")
    parser.add_argument(
        "--threshold", type=float, default=DEFAULT_THRESHOLD,
        help=f"Decision threshold for P(flare) → binary label "
             f"(default: {DEFAULT_THRESHOLD}).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
        help=f"Batch size for ViT inference (default: {DEFAULT_BATCH_SIZE}). "
             "Ignored for XGBoost (always batch_size=1).",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Output directory. Defaults to "
             "plots/aarp_eval/<run-id>/<subset-label>/.",
    )
    parser.add_argument(
        "--exclude-aarp", nargs="+", type=int, default=None,
        metavar="AARP_ID",
        help="One or more AARP IDs to exclude from evaluation "
             "(e.g. --exclude-aarp 1275 3153). Excluded AARPs are dropped "
             "after aggregation so inference still runs on the full dataset. "
             "The output directory gets an '_excl' suffix to avoid "
             "overwriting non-excluded results.",
    )
    args = parser.parse_args()

    run_id       = Path(args.model_path).parent.name
    subset_label = "+".join(args.subset)
    model_label  = f"{args.model_type}/{run_id}"

    excl_suffix = (
        "_excl" + "-".join(str(a) for a in sorted(args.exclude_aarp))
        if args.exclude_aarp else ""
    )

    if args.output_dir is None:
        args.output_dir = f"{DEFAULT_OUTPUT_DIR}/{run_id}/{subset_label}{excl_suffix}"

    print(f"Model      : {args.model_path}  [{args.model_type}]")
    print(f"Subset(s)  : {subset_label}")
    print(f"Threshold  : {args.threshold}")
    if args.exclude_aarp:
        print(f"Excluding  : {args.exclude_aarp}")
    print(f"Output dir : {args.output_dir}")

    # ── Load model ────────────────────────────────────────────────────────
    print("\nLoading model…")
    if args.model_type == "vit":
        model, transform, device = load_vit(args.model_path, args.stats_file)
        resize_to = None
        print(f"  Architecture : DeepFlare_ViT  |  device: {device}")
    elif args.model_type == "vit-pretrained":
        model, transform, device = load_vit_pretrained(args.model_path, args.stats_file)
        resize_to = 224
        print(f"  Architecture : torchvision vit_l_16  |  resize: 224  |  device: {device}")
    else:
        booster = load_xgb(args.model_path)
        print(f"  Architecture : XGBoost")

    # ── Get AARP IDs (needed for aggregation) ─────────────────────────────
    print("\nLoading metadata…")
    metadata = get_metadata_from_json(args.json_path)
    training_df, val_df, test_df = dfs_from_metadata(metadata)
    subset_df_map = {"training": training_df, "validation": val_df, "test": test_df}

    # ── Run inference ──────────────────────────────────────────────────────
    all_aarp_ids = []
    all_y_true   = []
    all_y_prob   = []

    for subset in args.subset:
        subset_df = subset_df_map[subset]
        print(f"\nProcessing '{subset}'  ({len(subset_df)} frames, "
              f"{subset_df['aarp_id'].nunique()} AARPs)…")

        if args.model_type in ("vit", "vit-pretrained"):
            dataset    = aia_euv(args.json_path, subset=subset,
                                 transform=v2.Compose([transform]))
            loader     = DataLoader(dataset, batch_size=args.batch_size,
                                    shuffle=False)
            y_true, y_prob = run_vit_inference(model, loader, device,
                                               resize_to=resize_to)
        else:
            y_true, y_prob = run_xgb_inference(
                booster, args.json_path, args.stats_file, subset
            )

        if len(y_true) != len(subset_df):
            raise RuntimeError(
                f"Prediction count ({len(y_true)}) does not match "
                f"DataFrame rows ({len(subset_df)}) for subset '{subset}'. "
                "Ensure shuffle=False and JSON order is preserved."
            )

        all_aarp_ids.extend(subset_df["aarp_id"].values)
        all_y_true.extend(y_true)
        all_y_prob.extend(y_prob)

    all_aarp_ids = np.array(all_aarp_ids)
    all_y_true   = np.array(all_y_true)
    all_y_prob   = np.array(all_y_prob)

    n_aarp = len(np.unique(all_aarp_ids))
    print(f"\nTotal: {len(all_y_true)} frames across {n_aarp} AARPs")

    # ── Aggregate & evaluate ───────────────────────────────────────────────
    agg = aggregate_per_aarp(all_aarp_ids, all_y_true, all_y_prob,
                              threshold=args.threshold)

    if args.exclude_aarp:
        before = len(agg)
        agg = agg[~agg["aarp_id"].isin(args.exclude_aarp)].reset_index(drop=True)
        dropped = before - len(agg)
        print(f"\nExcluded {dropped} AARP(s): {args.exclude_aarp}  "
              f"({len(agg)} AARPs remain)")

    # Per-AARP table to stdout
    pd.set_option("display.max_rows", None)
    pd.set_option("display.float_format", "{:.3f}".format)
    print("\nPer-AARP summary")
    print("=" * 72)
    print(agg[["aarp_id", "true_label", "n_frames",
               "max_prob", "mean_prob", "pred_max", "pred_mean"]].to_string(index=False))

    # Save outputs
    print("\nSaving outputs…")
    save_all_outputs(
        agg=agg,
        subset_label=subset_label,
        model_label=model_label,
        output_dir=args.output_dir,
        threshold=args.threshold,
        frame_y_true=all_y_true,
        frame_y_prob=all_y_prob,
    )


if __name__ == "__main__":
    main()
