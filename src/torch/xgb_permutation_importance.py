"""XGBoost Permutation Importance + TreeSHAP vs Gain Comparison.

Validates that XGBoost's built-in Gain importance reflects actual predictive
contribution by comparing it to two independent measures:

  1. Permutation Importance (sklearn) — shuffle each feature in the test set,
     measure recall drop. Measures real held-out predictive contribution.

  2. TreeSHAP (XGBoost built-in via pred_contribs=True) — theoretically-grounded
     per-sample SHAP values. No external 'shap' library required.

Note: captum.attr.KernelShap (used elsewhere in this codebase) is Captum's own
implementation of KernelSHAP for neural networks. It is unrelated to XGBoost's
TreeSHAP or the standalone 'shap' Python library.

Outputs written to --output-dir:
  perm_importance_top20.png    — top-20 by permutation importance (recall drop)
  gain_vs_perm_scatter.png     — Gain rank vs Perm rank scatter (Spearman ρ)
  rank_comparison_top20.png    — 3-way Gain / Permutation / TreeSHAP bar chart

Usage:
    python -m src.torch.xgb_permutation_importance \\
        --model-path outputs/devoted-pond-14/best_xgboost_model.json \\
        --output-dir plots/xgb/permutation_importance \\
        --n-repeats 10
"""

import argparse
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb
from scipy.stats import spearmanr
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

from aarp_ml.torch.dataset import AIALogTransform, aia_euv
from src.torch.xgb_train import TrainingConfig, extract_stats_generator, get_feature_names
from src.torch.vit.test import compute_metrics

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CONFUSION_MATRIX_CLASSES = [0, 1]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="XGBoost Permutation Importance + TreeSHAP vs Gain comparison"
    )
    parser.add_argument("--model-path",
                        default="outputs/devoted-pond-14/best_xgboost_model.json",
                        help="Path to saved XGBoost model JSON")
    parser.add_argument("--output-dir", default="plots/xgb/permutation_importance")
    parser.add_argument("--n-repeats", type=int, default=10,
                        help="Permutation importance repetitions per feature")
    parser.add_argument("--top-n", type=int, default=20,
                        help="Number of top features to display")
    parser.add_argument("--subset", default="test",
                        choices=["test", "validation", "training"])
    parser.add_argument("--json-path", default="solar_dataset.json")
    parser.add_argument("--stats-file", default="stats.pkl")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def build_transform(stats_file, n_channels=7):
    """Load normalization stats and return AIALogTransform."""
    with open(stats_file, "rb") as f:
        stats_data = pickle.load(f)
    means = [stats_data["mean"][f"channel_{i}"] for i in range(n_channels)]
    stds  = [stats_data["std"][f"channel_{i}"]  for i in range(n_channels)]
    return AIALogTransform(means=means, stds=stds)


def extract_test_features(json_path, stats_file, subset, config):
    """Extract statistical features for the test set.

    Uses extract_stats_generator() from xgb_train.py with batch_size=1 so
    that every sample is processed (the function takes images[0] per batch).

    Returns:
        X: np.ndarray [N, 77]
        y: np.ndarray [N]
    """
    transform = build_transform(stats_file)
    dataset = aia_euv(json_path, subset=subset, transform=transform)
    loader  = DataLoader(dataset, batch_size=1, shuffle=False)
    print(f"  Extracting features for {len(dataset)} {subset} samples...")
    X, y = extract_stats_generator(loader, config)
    print(f"  Feature matrix shape: {X.shape}")
    return X, y


# ---------------------------------------------------------------------------
# sklearn-compatible XGBoost wrapper
# ---------------------------------------------------------------------------

class XGBRecallWrapper:
    """Minimal sklearn-compatible wrapper around an XGBoost Booster.

    The score() method returns recall (TP / (TP + FN)) so that
    permutation_importance() measures the recall *drop* when a feature
    is shuffled — consistent with the project's existing evaluation metrics.
    """

    def __init__(self, booster, feature_names, threshold=0.5):
        self.booster = booster
        self.feature_names = feature_names
        self.threshold = threshold

    def fit(self, X, y):
        """No-op — model is already trained. Required by sklearn's API check."""
        return self

    def predict(self, X):
        dmat = xgb.DMatrix(X, feature_names=self.feature_names)
        probs = self.booster.predict(dmat)
        return (probs > self.threshold).astype(int)

    def score(self, X, y):
        preds = self.predict(X)
        tp = int(((preds == 1) & (y == 1)).sum())
        fn = int(((preds == 0) & (y == 1)).sum())
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0


# ---------------------------------------------------------------------------
# TreeSHAP via XGBoost built-in
# ---------------------------------------------------------------------------

def compute_treeshap(booster, X, feature_names):
    """Compute mean |TreeSHAP| per feature using XGBoost's built-in support.

    XGBoost natively supports TreeSHAP via pred_contribs=True (v0.90+).
    No external 'shap' library is required. The last column is the bias term
    and is dropped before averaging.

    Returns:
        mean_abs_shap: np.ndarray [n_features]
    """
    dmat = xgb.DMatrix(X, feature_names=feature_names)
    shap_values = booster.predict(dmat, pred_contribs=True)   # [N, n_features+1]
    shap_values = shap_values[:, :-1]                          # drop bias column → [N, n_features]
    return np.abs(shap_values).mean(axis=0)                    # [n_features]


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_perm_top20(perm_means, perm_stds, feature_names, top_n, output_dir):
    """Figure A: permutation importance — top-N by recall drop."""
    order = np.argsort(-perm_means)[:top_n]
    feat_labels = [feature_names[i] for i in order]
    vals  = perm_means[order]
    errs  = perm_stds[order]

    norm = plt.Normalize(vals.min(), vals.max())
    colours = plt.cm.viridis_r(norm(vals))

    fig, ax = plt.subplots(figsize=(13, 5))
    bars = ax.bar(range(top_n), vals, yerr=errs, capsize=4,
                  color=colours, alpha=0.85, edgecolor="white")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + vals.max() * 0.015,
                f"{v:.3f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(range(top_n))
    ax.set_xticklabels(feat_labels, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("Mean Recall Drop when shuffled (± std)")
    ax.set_title(f"XGBoost Permutation Importance — Top {top_n}\n"
                 "(Higher = feature more critical for recall on test set)", fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    out = os.path.join(output_dir, "perm_importance_top20.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


def plot_gain_vs_perm_scatter(gain_scores_dict, perm_means, feature_names, output_dir):
    """Figure B: scatter of Gain rank vs Permutation rank, with Spearman ρ."""
    n = len(feature_names)
    # Gain scores (0 for missing features)
    gain_arr = np.array([gain_scores_dict.get(f, 0.0) for f in feature_names])

    # Convert to ranks (1 = most important, n = least important)
    gain_rank = np.argsort(np.argsort(-gain_arr)) + 1   # 1-indexed: rank 1 = highest gain
    perm_rank = np.argsort(np.argsort(-perm_means)) + 1 # 1-indexed: rank 1 = highest recall drop

    rho = spearmanr(gain_rank, perm_rank).statistic

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(gain_rank, perm_rank, alpha=0.5, s=30, color="steelblue", zorder=3)

    # Label top-10 by gain (now at low rank numbers, near origin)
    top10_gain_idx = np.argsort(gain_arr)[-10:]
    for idx in top10_gain_idx:
        ax.annotate(feature_names[idx],
                    (gain_rank[idx], perm_rank[idx]),
                    fontsize=6, ha="left", va="bottom",
                    xytext=(3, 3), textcoords="offset points")

    # Reference diagonal
    lims = [1, n]
    ax.plot(lims, lims, "r--", alpha=0.4, linewidth=1, label="Perfect agreement")

    ax.set_xlabel("Gain Rank (1 = most important by Gain)")
    ax.set_ylabel("Permutation Rank (1 = most important by Recall Drop)")
    ax.set_title(f"Gain Rank vs Permutation Rank\nSpearman ρ = {rho:.3f}", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    out = os.path.join(output_dir, "gain_vs_perm_scatter.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")
    return rho


def plot_three_way_comparison(gain_scores_dict, perm_means, shap_means,
                               feature_names, top_n, output_dir):
    """Figure C: side-by-side bars — Gain / Permutation / TreeSHAP for top-N."""
    # Determine top-N by each method separately, then union
    gain_arr = np.array([gain_scores_dict.get(f, 0.0) for f in feature_names])
    top_by_gain = set(np.argsort(-gain_arr)[:top_n])
    top_by_perm = set(np.argsort(-perm_means)[:top_n])
    top_by_shap = set(np.argsort(-shap_means)[:top_n])
    top_idx = sorted(top_by_gain | top_by_perm | top_by_shap,
                     key=lambda i: gain_arr[i], reverse=True)[:top_n]
    top_features = [feature_names[i] for i in top_idx]

    # Normalise each method to [0, 1] for visual comparison
    def norm01(arr):
        r = arr.max() - arr.min()
        return (arr - arr.min()) / r if r > 0 else arr

    g_norm = norm01(gain_arr)[top_idx]
    p_norm = norm01(perm_means)[top_idx]
    s_norm = norm01(shap_means)[top_idx]

    x = np.arange(top_n)
    width = 0.27
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(x - width, g_norm, width, label="Gain (normalised)",        color="steelblue", alpha=0.8)
    ax.bar(x,          p_norm, width, label="Permutation (normalised)", color="darkorange", alpha=0.8)
    ax.bar(x + width,  s_norm, width, label="TreeSHAP (normalised)",   color="seagreen",  alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(top_features, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("Importance (normalised 0–1 within method)")
    ax.set_title(f"3-Way Importance Comparison — Top {top_n} by Gain\n"
                 "Agreement across all three → feature is genuinely important", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    out = os.path.join(output_dir, "rank_comparison_top20.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # -- Load XGBoost model --------------------------------------------------
    print(f"Loading model from {args.model_path}...")
    booster = xgb.Booster()
    booster.load_model(args.model_path)

    config = TrainingConfig(json_path=args.json_path, stats_file=args.stats_file)
    feature_names = get_feature_names(config)
    n_features = len(feature_names)

    gain_scores = booster.get_score(importance_type="gain")
    print(f"  Model has {n_features} features; {len(gain_scores)} with non-zero Gain")

    # -- Extract test features -----------------------------------------------
    print(f"\nExtracting features for '{args.subset}' set...")
    X_test, y_test = extract_test_features(
        args.json_path, args.stats_file, args.subset, config
    )

    # Baseline evaluation (no masking)
    wrapper = XGBRecallWrapper(booster, feature_names)
    baseline_recall = wrapper.score(X_test, y_test)
    preds_base = wrapper.predict(X_test)
    cm_base = confusion_matrix(y_test, preds_base, labels=CONFUSION_MATRIX_CLASSES)
    base_metrics = compute_metrics(cm_base)
    print(f"\n  Baseline ({args.subset}):")
    print(f"    Recall:    {base_metrics['recall']:.4f}")
    print(f"    Precision: {base_metrics['precision']:.4f}")

    # -- TreeSHAP ------------------------------------------------------------
    print("\nComputing TreeSHAP (XGBoost built-in pred_contribs=True)...")
    shap_means = compute_treeshap(booster, X_test, feature_names)
    top5_shap = np.argsort(-shap_means)[:5]
    print("  Top-5 by mean |TreeSHAP|:")
    for i in top5_shap:
        print(f"    {feature_names[i]:<15}  {shap_means[i]:.5f}")

    # -- Permutation importance ----------------------------------------------
    print(f"\nRunning permutation importance (n_repeats={args.n_repeats})...")
    print("  (This shuffles each feature and measures recall drop — may take a while)")
    perm_result = permutation_importance(
        wrapper, X_test, y_test,
        n_repeats=args.n_repeats,
        random_state=42,
        n_jobs=1,       # sequential to avoid conflicts with XGBoost threading
    )
    perm_means = perm_result.importances_mean   # [n_features] recall drop
    perm_stds  = perm_result.importances_std

    top5_perm = np.argsort(-perm_means)[:5]
    print("  Top-5 by permutation importance (recall drop):")
    for i in top5_perm:
        print(f"    {feature_names[i]:<15}  drop={perm_means[i]:.4f} ± {perm_stds[i]:.4f}")

    # -- Compare rankings (Spearman ρ) ---------------------------------------
    gain_arr = np.array([gain_scores.get(f, 0.0) for f in feature_names])
    rho_gain_perm = spearmanr(
        np.argsort(np.argsort(-gain_arr)),
        np.argsort(np.argsort(-perm_means))
    ).statistic
    rho_gain_shap = spearmanr(
        np.argsort(np.argsort(-gain_arr)),
        np.argsort(np.argsort(-shap_means))
    ).statistic
    rho_perm_shap = spearmanr(
        np.argsort(np.argsort(-perm_means)),
        np.argsort(np.argsort(-shap_means))
    ).statistic

    print("\n--- Rank Correlation Summary (Spearman ρ, higher = more agreement) ---")
    print(f"  Gain vs Permutation:  ρ = {rho_gain_perm:.3f}")
    print(f"  Gain vs TreeSHAP:     ρ = {rho_gain_shap:.3f}")
    print(f"  Permutation vs SHAP:  ρ = {rho_perm_shap:.3f}")

    if rho_gain_perm > 0.6:
        print("  ✅ Gain ranking agrees with Permutation (ρ > 0.6) — Gain is reliable")
    elif rho_gain_perm > 0.3:
        print("  ⚠️  Partial agreement (0.3 < ρ ≤ 0.6) — interpret Gain with caution")
    else:
        print("  ❌ Low agreement (ρ ≤ 0.3) — Gain ranking may be misleading")

    # -- Figures -------------------------------------------------------------
    print("\nGenerating figures...")
    plot_perm_top20(perm_means, perm_stds, feature_names, args.top_n, args.output_dir)
    plot_gain_vs_perm_scatter(gain_scores, perm_means, feature_names, args.output_dir)
    plot_three_way_comparison(gain_scores, perm_means, shap_means,
                              feature_names, args.top_n, args.output_dir)

    print(f"\nOutputs written to: {args.output_dir}/")


if __name__ == "__main__":
    main()
