"""XGBoost Cross-Run Rank Consistency Test.

Validates that the Gain-based feature importance ranking is stable across the
three saved XGBoost runs. No data loading required — works entirely from the
saved model JSON files.

Computes pairwise Kendall's τ between runs on shared features, a consensus
ranking (mean rank across runs), and a spaghetti plot showing each feature's
rank in each run.

Outputs written to --output-dir:
  pairwise_kendall_tau.png   — 3×3 heatmap of τ between all run pairs
  consensus_rank_top20.png   — top-20 by mean rank, bars = mean gain ± std
  rank_stability.png         — per-feature rank across runs (spaghetti, top-20)

Usage:
    python -m src.torch.xgb_consistency_test \\
        --output-dir plots/xgb/consistency_test
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb
from scipy.stats import kendalltau

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_PATHS = {
    "apricot-fog-13":    "outputs/apricot-fog-13/best_xgboost_model.json",
    "deep-spaceship-10": "outputs/deep-spaceship-10/best_xgboost_model.json",
    "devoted-pond-14":   "outputs/devoted-pond-14/best_xgboost_model.json",
}

RUN_COLOURS = {
    "apricot-fog-13":    "steelblue",
    "deep-spaceship-10": "darkorange",
    "devoted-pond-14":   "seagreen",
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="XGBoost Cross-Run Feature Importance Rank Consistency"
    )
    parser.add_argument("--output-dir", default="plots/xgb/consistency_test")
    parser.add_argument("--top-n", type=int, default=20,
                        help="Number of top features to display in consensus/spaghetti plots")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Load importance scores from saved model JSONs
# ---------------------------------------------------------------------------

def load_gain_scores(model_path):
    """Load a saved XGBoost model and return its Gain importance dict.

    Returns {feature_name: gain_score} for features that appeared in splits.
    Features with zero gain are not returned by XGBoost (they are absent from
    the dict), which is accounted for in downstream alignment.
    """
    booster = xgb.Booster()
    booster.load_model(model_path)
    return booster.get_score(importance_type="gain")


def align_scores(scores_by_run):
    """Align gain scores across runs on the *union* of all features.

    Missing features in a run get score 0.0.
    Returns a dict {feature: {run_name: score}}.
    """
    all_features = set()
    for scores in scores_by_run.values():
        all_features.update(scores.keys())

    aligned = {}
    for feat in all_features:
        aligned[feat] = {
            run: scores_by_run[run].get(feat, 0.0)
            for run in scores_by_run
        }
    return aligned


def scores_to_rank_array(scores_dict, run_names, all_features):
    """Convert {feature: score} to a rank array aligned to all_features.

    Rank 1 = most important. Features absent from the model get rank = last+1.
    Returns np.ndarray [n_features].
    """
    n = len(all_features)
    # Build score vector (0 for missing)
    score_vec = np.array([scores_dict.get(f, 0.0) for f in all_features])
    # Rank: argsort descending, then assign 1-based ranks
    order = np.argsort(-score_vec)          # indices from most to least important
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(1, n + 1)
    return ranks


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def pairwise_kendall(rank_arrays, run_names):
    """Return n_runs × n_runs Kendall's τ matrix."""
    n = len(run_names)
    mat = np.ones((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                tau = kendalltau(rank_arrays[i], rank_arrays[j]).statistic
                mat[i, j] = tau
    return mat


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_tau_heatmap(tau_mat, run_names, output_dir):
    """Figure A: pairwise Kendall's τ heatmap."""
    n = len(run_names)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(tau_mat, cmap="coolwarm", vmin=-1, vmax=1)
    fig.colorbar(im, ax=ax, label="Kendall's τ")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    short_names = [r.split("-")[0] + "-" + r.split("-")[1] for r in run_names]
    ax.set_xticklabels(short_names, rotation=30, ha="right", fontsize=9)
    ax.set_yticklabels(short_names, fontsize=9)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{tau_mat[i, j]:.2f}", ha="center", va="center",
                    fontsize=10, fontweight="bold",
                    color="white" if abs(tau_mat[i, j]) > 0.6 else "black")
    ax.set_title("Pairwise Kendall's τ\n(Gain importance rank agreement across runs)",
                 fontsize=10)
    plt.tight_layout()
    out = os.path.join(output_dir, "pairwise_kendall_tau.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


def plot_consensus_top20(aligned, run_names, top_n, output_dir):
    """Figure B: top-N features by mean rank, bar height = mean gain ± std."""
    # Compute mean rank and mean gain across runs for each feature
    all_features = list(aligned.keys())
    n_features = len(all_features)

    mean_gains = np.array([
        np.mean([aligned[f][r] for r in run_names]) for f in all_features
    ])
    std_gains = np.array([
        np.std([aligned[f][r] for r in run_names]) for f in all_features
    ])

    # Sort by mean gain descending, take top_n
    order = np.argsort(-mean_gains)[:top_n]
    top_features  = [all_features[i] for i in order]
    top_mean_gain = mean_gains[order]
    top_std_gain  = std_gains[order]

    # Colour by mean gain magnitude (viridis_r)
    norm = plt.Normalize(top_mean_gain.min(), top_mean_gain.max())
    colours = plt.cm.viridis_r(norm(top_mean_gain))

    fig, ax = plt.subplots(figsize=(13, 5))
    bars = ax.bar(range(top_n), top_mean_gain, yerr=top_std_gain,
                  capsize=4, color=colours, alpha=0.85, edgecolor="white")
    for bar, val in zip(bars, top_mean_gain):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + top_mean_gain.max() * 0.015,
                f"{val:.1f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(range(top_n))
    ax.set_xticklabels(top_features, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("Mean Gain across runs (± std)")
    ax.set_title(f"XGBoost Consensus Ranking — Top {top_n} by Mean Gain\n"
                 f"({len(run_names)} runs: {', '.join(run_names)})", fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    out = os.path.join(output_dir, "consensus_rank_top20.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


def plot_rank_stability(rank_arrays, aligned, run_names, top_n, output_dir):
    """Figure C: spaghetti plot — each feature's rank across runs (top-N by consensus)."""
    all_features = list(aligned.keys())
    mean_gains = np.array([
        np.mean([aligned[f][r] for r in run_names]) for f in all_features
    ])
    order = np.argsort(-mean_gains)[:top_n]
    top_features = [all_features[i] for i in order]

    # Build rank matrix: [n_runs, top_n]
    rank_mat = np.array([
        [rank_arrays[ri][oi] for oi in order]
        for ri in range(len(run_names))
    ])

    fig, ax = plt.subplots(figsize=(13, 5))
    x = np.arange(top_n)
    for ri, (run, colour) in enumerate(zip(run_names, RUN_COLOURS.values())):
        ax.plot(x, rank_mat[ri], "o-", color=colour, linewidth=1.5,
                markersize=6, label=run, alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(top_features, rotation=40, ha="right", fontsize=8)
    ax.invert_yaxis()   # Rank 1 at top
    ax.set_ylabel("Rank (1 = most important)")
    ax.set_title(f"Feature Rank per Run — Top {top_n} by Consensus\n"
                 f"Flat lines = stable ranking; crossings = instability", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    out = os.path.join(output_dir, "rank_stability.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # -- Load gain scores from all runs --------------------------------------
    print("Loading Gain importance from saved model JSONs...")
    scores_by_run = {}
    for run_name, model_path in MODEL_PATHS.items():
        if not os.path.exists(model_path):
            print(f"  ⚠  {run_name}: model not found at {model_path}, skipping")
            continue
        scores = load_gain_scores(model_path)
        scores_by_run[run_name] = scores
        print(f"  {run_name}: {len(scores)} features with non-zero gain")

    if len(scores_by_run) < 2:
        raise RuntimeError("Need at least 2 runs for comparison.")

    run_names = list(scores_by_run.keys())

    # -- Align on union of all features -------------------------------------
    aligned = align_scores(scores_by_run)
    all_features = list(aligned.keys())
    print(f"\nTotal unique features across runs: {len(all_features)}")

    # -- Rank arrays ---------------------------------------------------------
    rank_arrays = [
        scores_to_rank_array(scores_by_run[r], run_names, all_features)
        for r in run_names
    ]

    # -- Pairwise Kendall's τ ------------------------------------------------
    tau_mat = pairwise_kendall(rank_arrays, run_names)
    print("\n--- Pairwise Kendall's τ (on all shared features) ---")
    for i, r1 in enumerate(run_names):
        for j, r2 in enumerate(run_names):
            if j > i:
                print(f"  {r1} vs {r2}: τ = {tau_mat[i, j]:.3f}")

    # -- Top feature summary -------------------------------------------------
    mean_gains = {
        f: np.mean([aligned[f][r] for r in run_names])
        for f in all_features
    }
    top10 = sorted(mean_gains.items(), key=lambda x: x[1], reverse=True)[:10]
    print("\n--- Consensus Top-10 by Mean Gain ---")
    for rank, (feat, gain) in enumerate(top10, 1):
        per_run = [f"{r}: {aligned[feat][r]:.1f}" for r in run_names]
        print(f"  {rank:2d}. {feat:<15}  mean={gain:.1f}  ({', '.join(per_run)})")

    # -- Plots ---------------------------------------------------------------
    print("\nGenerating figures...")
    plot_tau_heatmap(tau_mat, run_names, args.output_dir)
    plot_consensus_top20(aligned, run_names, args.top_n, args.output_dir)
    plot_rank_stability(rank_arrays, aligned, run_names, args.top_n, args.output_dir)

    print(f"\nOutputs written to: {args.output_dir}/")


if __name__ == "__main__":
    main()
