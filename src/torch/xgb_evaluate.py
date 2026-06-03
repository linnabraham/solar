"""XGBoost evaluation script — classification metrics for paper reporting.

Loads a saved XGBoost model, extracts per-frame statistical features from the
requested split(s), and reports standard binary-classification metrics plus an
optional solar-flare skill score.

Core metrics (always computed)
-------------------------------
  Confusion matrix  — TP, FP, FN, TN
  Precision         — TP / (TP + FP)
  Recall            — TP / (TP + FN)
  FAR               — FP / (FP + TN)   (False Alarm Rate)
  F1                — harmonic mean of precision and recall
  Accuracy          — (TP + TN) / N
  Log-loss (BCE)    — cross-entropy on raw probabilities

Optional skill score (--tss flag)
----------------------------------
  TSS = Recall − FAR

  TSS is widely cited in solar-flare prediction papers even for snapshot
  classifiers, but note the caveat: time information is used only during
  data selection (positive = images prior to flare), not as a model input.
  The metric is therefore a function of the confusion matrix only, with no
  genuine lead-time interpretation. Enable it explicitly so the choice is
  documented.

Outputs
-------
  Metrics table       → stdout
  Confusion matrix    → <output-dir>/cm_xgb_<subset>.png
  Metrics CSV         → <output-dir>/xgb_metrics.csv  (one row per subset)

Usage
-----
    # core metrics on test set
    python -m src.torch.xgb_evaluate \\
        --model-path outputs/devoted-pond-14/best_xgboost_model.json \\
        --subset test

    # all splits + TSS
    python -m src.torch.xgb_evaluate \\
        --model-path outputs/devoted-pond-14/best_xgboost_model.json \\
        --subset training validation test \\
        --tss
"""

import argparse
import csv
import os
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb
from sklearn.metrics import confusion_matrix, log_loss
from torch.utils.data import DataLoader

from aarp_ml.torch.dataset import AIALogTransform, aia_euv
from ml_utils.visualization import plot_confusion_matrix
from src.torch.xgb_train import TrainingConfig, extract_stats_generator, get_feature_names

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CONFUSION_MATRIX_CLASSES = [0, 1]
VALID_SUBSETS = {"training", "validation", "test"}

DEFAULT_MODEL_PATH = "outputs/devoted-pond-14/best_xgboost_model.json"
DEFAULT_JSON_PATH  = "solar_dataset.json"
DEFAULT_STATS_FILE = "stats.pkl"
DEFAULT_OUTPUT_DIR = "plots/xgb/evaluation"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class EvalConfig:
    model_path: str = DEFAULT_MODEL_PATH
    json_path:  str = DEFAULT_JSON_PATH
    stats_file: str = DEFAULT_STATS_FILE
    output_dir: str = DEFAULT_OUTPUT_DIR
    subsets:    List[str] = field(default_factory=lambda: ["test"])
    threshold:  float = 0.5
    n_channels: int = 7
    include_tss: bool = False   # opt-in — see module docstring for the caveat


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def _build_transform(stats_file: str, n_channels: int) -> AIALogTransform:
    with open(stats_file, "rb") as f:
        stats_data = pickle.load(f)
    means = [stats_data["mean"][f"channel_{i}"] for i in range(n_channels)]
    stds  = [stats_data["std"][f"channel_{i}"]  for i in range(n_channels)]
    return AIALogTransform(means=means, stds=stds)


def extract_features(
    json_path: str,
    stats_file: str,
    subset: str,
    config: TrainingConfig,
    n_channels: int = 7,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (X, y) for *subset*. batch_size=1 so every frame is captured."""
    transform = _build_transform(stats_file, n_channels)
    dataset   = aia_euv(json_path, subset=subset, transform=transform)
    loader    = DataLoader(dataset, batch_size=1, shuffle=False)
    print(f"  Extracting features: {len(dataset)} frames in '{subset}' split …")
    X, y = extract_stats_generator(loader, config)
    print(f"  Feature matrix: {X.shape}")
    return X, y


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

def compute_core_metrics(
    cm: np.ndarray,
    probs: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, float]:
    """Standard binary-classification metrics from a 2×2 confusion matrix."""
    TN, FP, FN, TP = cm.ravel()
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    far       = FP / (FP + TN) if (FP + TN) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    accuracy  = (TP + TN) / len(labels)
    bce       = log_loss(labels, probs)
    return {
        "TP": int(TP), "FP": int(FP), "FN": int(FN), "TN": int(TN),
        "precision": precision,
        "recall":    recall,
        "FAR":       far,
        "F1":        f1,
        "accuracy":  accuracy,
        "log_loss":  bce,
    }


# ---------------------------------------------------------------------------
# Optional skill score
# ---------------------------------------------------------------------------

def compute_tss(metrics: Dict[str, float]) -> float:
    """True Skill Score = Recall − FAR.

    Widely cited in solar-flare literature but carries a caveat for this
    model: time is used only for data selection, not as a model input, so
    TSS here is a confusion-matrix statistic with no lead-time interpretation.
    """
    return metrics["recall"] - metrics["FAR"]


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def print_metrics(subset: str, m: Dict[str, float], tss: Optional[float]) -> None:
    sep = "=" * 56
    print(f"\n{sep}")
    print(f"  XGBoost — {subset} set")
    print(sep)
    print(f"  TP: {m['TP']:4d}   FP: {m['FP']:4d}")
    print(f"  FN: {m['FN']:4d}   TN: {m['TN']:4d}")
    print(f"  Precision : {m['precision']:.4f}")
    print(f"  Recall    : {m['recall']:.4f}")
    print(f"  FAR       : {m['FAR']:.4f}")
    print(f"  F1        : {m['F1']:.4f}")
    print(f"  Accuracy  : {m['accuracy']:.4f}")
    print(f"  Log-loss  : {m['log_loss']:.4f}")
    if tss is not None:
        print(f"  TSS       : {tss:.4f}  [Recall − FAR; see --tss caveat]")
    print(sep)


def save_confusion_matrix(cm: np.ndarray, subset: str, output_dir: str) -> None:
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
    out = Path(output_dir) / f"cm_xgb_{subset}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  ✓ Confusion matrix → {out}")


def save_metrics_csv(rows: List[Dict], output_dir: str) -> None:
    """Write one row per evaluated (model, subset) to a CSV."""
    out = Path(output_dir) / "xgb_metrics.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    # collect all keys in order, TSS last if present
    base_keys = ["model", "subset",
                 "TP", "FP", "FN", "TN",
                 "precision", "recall", "FAR", "F1", "accuracy", "log_loss"]
    has_tss   = any("TSS" in r for r in rows)
    fieldnames = base_keys + (["TSS"] if has_tss else [])
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n✓ Metrics CSV → {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(cfg: EvalConfig) -> None:
    os.makedirs(cfg.output_dir, exist_ok=True)

    print(f"Loading model: {cfg.model_path}")
    booster = xgb.Booster()
    booster.load_model(cfg.model_path)

    train_cfg     = TrainingConfig(json_path=cfg.json_path, stats_file=cfg.stats_file)
    feature_names = get_feature_names(train_cfg)
    model_name    = Path(cfg.model_path).parent.name
    csv_rows: List[Dict] = []

    # Accumulate across subsets for optional combined CM
    all_y: List[np.ndarray]     = []
    all_probs: List[np.ndarray] = []

    for subset in cfg.subsets:
        print(f"\n── Evaluating '{subset}' ─────────────────────────")
        X, y = extract_features(
            cfg.json_path, cfg.stats_file, subset, train_cfg, cfg.n_channels
        )

        dmat  = xgb.DMatrix(X, label=y, feature_names=feature_names)
        probs = booster.predict(dmat)
        preds = (probs > cfg.threshold).astype(int)

        all_y.append(y)
        all_probs.append(probs)

        cm = confusion_matrix(y, preds, labels=CONFUSION_MATRIX_CLASSES)
        m  = compute_core_metrics(cm, probs, y)
        tss = compute_tss(m) if cfg.include_tss else None

        print_metrics(subset, m, tss)
        save_confusion_matrix(cm, subset, cfg.output_dir)

        row = {"model": model_name, "subset": subset, **m}
        if tss is not None:
            row["TSS"] = tss
        csv_rows.append(row)

    # Combined CM when multiple subsets given
    if len(cfg.subsets) > 1:
        combined_label = "+".join(cfg.subsets)
        print(f"\n── Combined '{combined_label}' ──────────────────────")
        y_comb     = np.concatenate(all_y)
        probs_comb = np.concatenate(all_probs)
        preds_comb = (probs_comb > cfg.threshold).astype(int)

        cm_comb = confusion_matrix(y_comb, preds_comb, labels=CONFUSION_MATRIX_CLASSES)
        m_comb  = compute_core_metrics(cm_comb, probs_comb, y_comb)
        tss_comb = compute_tss(m_comb) if cfg.include_tss else None

        print_metrics(combined_label, m_comb, tss_comb)
        save_confusion_matrix(cm_comb, combined_label, cfg.output_dir)

        row = {"model": model_name, "subset": combined_label, **m_comb}
        if tss_comb is not None:
            row["TSS"] = tss_comb
        csv_rows.append(row)

    save_metrics_csv(csv_rows, cfg.output_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a saved XGBoost model and report paper metrics.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model-path", default=DEFAULT_MODEL_PATH,
        help="Path to saved XGBoost model JSON (default: devoted-pond-14)",
    )
    parser.add_argument(
        "--subset", nargs="+", default=["test"],
        choices=list(VALID_SUBSETS), dest="subsets",
        help="Split(s) to evaluate. Default: test",
    )
    parser.add_argument("--json-path",  default=DEFAULT_JSON_PATH)
    parser.add_argument("--stats-file", default=DEFAULT_STATS_FILE)
    parser.add_argument(
        "--output-dir", default=None,
        help="Directory for confusion matrix PNGs and metrics CSV. "
             "Defaults to plots/xgb/evaluation/<run-id>/.",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Decision threshold for P(flare) → binary label (default: 0.5)",
    )
    parser.add_argument(
        "--tss", action="store_true", default=False,
        help=(
            "Also compute TSS = Recall − FAR. "
            "Note: time is only used for data selection, not as a model input — "
            "TSS here is a confusion-matrix statistic with no lead-time meaning."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.output_dir is None:
        run_id = Path(args.model_path).parent.name
        args.output_dir = f"{DEFAULT_OUTPUT_DIR}/{run_id}"
    cfg = EvalConfig(
        model_path=args.model_path,
        json_path=args.json_path,
        stats_file=args.stats_file,
        output_dir=args.output_dir,
        subsets=args.subsets,
        threshold=args.threshold,
        include_tss=args.tss,
    )
    main(cfg)
