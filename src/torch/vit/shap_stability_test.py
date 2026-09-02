"""SHAP Stability Test — KernelSHAP stochasticity check.

KernelSHAP randomly samples feature coalitions. This script runs it multiple
times on the same inputs (different seeds per run) and measures whether the
7-channel importance *rankings* are stable across runs.

If rankings are unstable, downstream faithfulness claims are unreliable.

Outputs written to --output-dir:
  channel_importance_variability.png  — boxplot of importance per channel,
                                        stratified by class, across all runs
  kendall_tau_heatmap.png             — pairwise Kendall's τ between runs
  cv_per_channel.png                  — coefficient of variation per channel

Usage:
    python -m src.torch.vit.shap_stability_test \\
        --output-dir plots/shap_stability_test \\
        --n-samples 6 --n-runs 5 --n-shap 100 --subset test
"""

import argparse
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from captum.attr import KernelShap
from scipy.stats import kendalltau

from torchvision.transforms import v2

from src.torch.vit.ig import single_aarp
from src.torch.vit.train import TrainingConfig
from src.torch.vit.utils import (
    dfs_from_metadata,
    get_metadata_from_json,
    get_model_and_transform,
    needs_resize,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TRAINED_MODEL_PATH = "outputs/glad-shape-197/trained_model.pth"
AIA_WAVELENGTHS = [94, 131, 171, 193, 211, 304, 335]
CHANNEL_LABELS = [f"{w} Å" for w in AIA_WAVELENGTHS]
N_CHANNELS = 7
VALID_MODEL_TYPES = {"vit": "deepflare_vit", "vit-pretrained": "vit_pretrained"}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="SHAP Stability Test — measures KernelSHAP reproducibility"
    )
    parser.add_argument("--model-path", default=TRAINED_MODEL_PATH,
                        help="Path to trained ViT model checkpoint.")
    parser.add_argument("--output-dir", default=None,
                        help="Directory for output plots. Defaults to "
                             "plots/shap_stability_test/<run-id> derived from --model-path.")
    parser.add_argument("--n-samples", type=int, default=6,
                        help="Samples to use (balanced: up to n/2 flare + n/2 non-flare)")
    parser.add_argument("--n-runs", type=int, default=5,
                        help="Repeated KernelSHAP runs per sample (different seeds)")
    parser.add_argument("--n-shap", type=int, default=100,
                        help="KernelSHAP n_samples parameter (coalitions). Min=25 for speed")
    parser.add_argument("--subset", default="test",
                        choices=["test", "validation", "training"])
    parser.add_argument("--json-path", default="solar_dataset.json")
    parser.add_argument("--stats-file", default="stats.pkl")
    parser.add_argument("--model-type", default="vit", choices=list(VALID_MODEL_TYPES),
                        help="Model architecture: 'vit' (DeepFlare_ViT, default) or "
                             "'vit-pretrained' (torchvision vit_l_16).")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def pick_balanced_samples(df, n_total):
    """Return up to n_total AARP single-timestep images, class-balanced."""
    samples = []
    per_class = n_total // 2
    for label in [1, 0]:
        aarp_ids = df[df["label"] == label]["aarp_id"].unique()
        for aarp_id in aarp_ids:
            if sum(1 for _, lbl in samples if lbl == label) >= per_class:
                break
            aarp_df = df[df["aarp_id"] == int(aarp_id)]
            s = single_aarp(int(aarp_id), aarp_df)
            try:
                images = s.get_images()          # [T, 7, 512, 512]
            except Exception as e:
                print(f"  Skipping AARP {aarp_id}: {e}")
                continue
            mid = len(images) // 2
            samples.append((images[mid], int(s.label)))   # ([7,512,512], int)
    return samples


# ---------------------------------------------------------------------------
# Single KernelSHAP call with controlled seed
# ---------------------------------------------------------------------------

def compute_shap_one_run(image_np, transform, baseline_per_channel,
                         model, device, n_shap, seed, resize_to=None):
    """Run KernelSHAP once with a fixed seed. Returns importance array [7]."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    tensor_img = torch.from_numpy(image_np).float()
    transformed = transform(tensor_img).unsqueeze(0)   # [1, 7, 512, 512]
    if resize_to is not None:
        transformed = v2.Resize(resize_to)(transformed)
    transformed = transformed.to(device)

    # Build baseline tensor: channel c → baseline_per_channel[c] everywhere
    baseline = torch.zeros_like(transformed)
    for c in range(N_CHANNELS):
        baseline[:, c, :, :] = baseline_per_channel[c]

    # Feature mask: all pixels in channel c share feature id c
    feature_mask = torch.zeros_like(transformed, dtype=torch.long)
    for c in range(N_CHANNELS):
        feature_mask[:, c, :, :] = c

    model.eval()
    explainer = KernelShap(model)
    attrs = explainer.attribute(
        transformed,
        baselines=baseline,
        feature_mask=feature_mask,
        n_samples=n_shap,
        target=1,               # Explain w.r.t. flare class (consistent with kshap.py)
        show_progress=False,
    )
    # Mean over batch, H, W → [7]
    return attrs.mean(dim=(0, 2, 3)).detach().cpu().numpy()


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def all_pairs_kendall(runs_matrix):
    """Compute mean Kendall's τ over all ordered pairs of runs.

    Args:
        runs_matrix: np.ndarray [n_runs, 7]

    Returns:
        pairwise_matrix [n_runs, n_runs], mean_tau (float)
    """
    n = runs_matrix.shape[0]
    mat = np.ones((n, n))
    taus = []
    for i in range(n):
        for j in range(n):
            if i != j:
                tau = kendalltau(runs_matrix[i], runs_matrix[j]).statistic
                mat[i, j] = tau
                taus.append(tau)
    return mat, float(np.mean(taus)) if taus else 1.0


def cv_per_channel(runs_matrix):
    """Coefficient of variation per channel across runs. [7]"""
    means = np.abs(runs_matrix.mean(axis=0))   # |mean| avoids sign cancellation
    stds = runs_matrix.std(axis=0)
    # Avoid division by zero; set CV=0 when mean≈0
    cv = np.where(means > 1e-10, stds / means, 0.0)
    return cv


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_variability(all_results, output_dir):
    """Figure A: boxplot of importance per channel, stratified by class."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
    class_names = {0: "Non-flare", 1: "Flare"}

    # Sort channels by global mean |SHAP| for display
    all_importances = np.vstack([r["runs"] for r in all_results])  # [N_samples*N_runs, 7]
    global_mean_abs = np.abs(all_importances).mean(axis=0)
    sorted_ch_idx = np.argsort(global_mean_abs)[::-1]   # descending
    sorted_labels = [CHANNEL_LABELS[i] for i in sorted_ch_idx]

    for col, label_val in enumerate([1, 0]):
        ax = axes[col]
        data_by_channel = []
        for ch_idx in sorted_ch_idx:
            vals = np.concatenate([r["runs"][:, ch_idx]
                                   for r in all_results if r["label"] == label_val])
            data_by_channel.append(vals)

        bp = ax.boxplot(data_by_channel, labels=sorted_labels, patch_artist=True,
                        medianprops=dict(color="black", linewidth=1.5))
        colours = plt.cm.tab10(np.linspace(0, 0.9, N_CHANNELS))
        for patch, colour in zip(bp["boxes"], colours):
            patch.set_facecolor(colour)
            patch.set_alpha(0.6)

        # Overlay individual run points (jittered)
        for i, ch_idx in enumerate(sorted_ch_idx):
            for r in all_results:
                if r["label"] != label_val:
                    continue
                jitter = np.random.uniform(-0.15, 0.15, size=r["runs"].shape[0])
                ax.scatter(np.full(r["runs"].shape[0], i + 1) + jitter,
                           r["runs"][:, ch_idx], s=15, alpha=0.5, color="grey", zorder=3)

        ax.axhline(0, color="red", linestyle="--", alpha=0.4, linewidth=0.8)
        ax.set_title(f"{class_names[label_val]}", fontsize=11)
        ax.set_xlabel("AIA Channel (sorted by mean |SHAP|)")
        ax.set_ylabel("SHAP Importance")
        ax.tick_params(axis="x", rotation=20)

    fig.suptitle(
        f"KernelSHAP Stability — Channel Importance Variability Across Runs",
        fontsize=12,
    )
    plt.tight_layout()
    out = os.path.join(output_dir, "channel_importance_variability.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


def plot_kendall_heatmap(all_results, n_runs, output_dir):
    """Figure B: pairwise Kendall's τ heatmap, averaged across samples."""
    # Average τ matrices across all samples
    avg_mat = np.zeros((n_runs, n_runs))
    count = 0
    for r in all_results:
        mat, _ = all_pairs_kendall(r["runs"])
        avg_mat += mat
        count += 1
    avg_mat /= max(count, 1)

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(avg_mat, cmap="coolwarm", vmin=-1, vmax=1)
    fig.colorbar(im, ax=ax, label="Kendall's τ")
    ax.set_xticks(range(n_runs))
    ax.set_yticks(range(n_runs))
    ax.set_xticklabels([f"Run {i}" for i in range(n_runs)], fontsize=8)
    ax.set_yticklabels([f"Run {i}" for i in range(n_runs)], fontsize=8)
    for i in range(n_runs):
        for j in range(n_runs):
            ax.text(j, i, f"{avg_mat[i, j]:.2f}", ha="center", va="center",
                    fontsize=8, color="black")
    ax.set_title("Pairwise Kendall's τ (avg over samples)\n"
                 "Diagonal=1.0 (self), off-diagonal = cross-run agreement", fontsize=9)
    plt.tight_layout()
    out = os.path.join(output_dir, "kendall_tau_heatmap.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


def plot_cv(all_results, output_dir):
    """Figure C: coefficient of variation per channel, averaged across samples."""
    # CV per channel, averaged over all samples
    cvs = np.vstack([cv_per_channel(r["runs"]) for r in all_results])  # [n_samples, 7]
    mean_cv = cvs.mean(axis=0)
    std_cv = cvs.std(axis=0)

    # Sort by mean |SHAP| for display
    all_importances = np.vstack([r["runs"] for r in all_results])
    sorted_idx = np.argsort(np.abs(all_importances).mean(axis=0))[::-1]
    sorted_labels = [CHANNEL_LABELS[i] for i in sorted_idx]
    sorted_cv_mean = mean_cv[sorted_idx]
    sorted_cv_std = std_cv[sorted_idx]

    fig, ax = plt.subplots(figsize=(8, 4))
    colours = plt.cm.tab10(np.linspace(0, 0.9, N_CHANNELS))
    bars = ax.bar(range(N_CHANNELS), sorted_cv_mean,
                  yerr=sorted_cv_std, capsize=4,
                  color=[colours[i] for i in sorted_idx], alpha=0.8)
    ax.axhline(0.5, color="red", linestyle="--", alpha=0.6,
               label="CV = 0.5 (high instability)")
    ax.set_xticks(range(N_CHANNELS))
    ax.set_xticklabels(sorted_labels, rotation=20)
    ax.set_ylabel("Coefficient of Variation (std / |mean|)")
    ax.set_xlabel("AIA Channel (sorted by mean |SHAP|)")
    ax.set_title("KernelSHAP Stability — CV per Channel\n"
                 "Lower CV = more stable importance across runs")
    ax.legend(fontsize=9)
    plt.tight_layout()
    out = os.path.join(output_dir, "cv_per_channel.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    from pathlib import Path
    args = parse_args()
    if args.output_dir is None:
        run_id = Path(args.model_path).parent.name
        args.output_dir = f"plots/shap_stability_test/{run_id}"
    os.makedirs(args.output_dir, exist_ok=True)

    # -- Load model + data ---------------------------------------------------
    print("Loading model and data...")
    config = TrainingConfig(
        json_path=args.json_path,
        stats_file=args.stats_file,
        trained_model_path=args.model_path,
        model_type=VALID_MODEL_TYPES[args.model_type],
    )
    model, transform, device = get_model_and_transform(config)
    resize_to = needs_resize(config)
    print(f"Model type: {args.model_type}" + (f"  (resize to {resize_to})" if resize_to else ""))

    metadata = get_metadata_from_json(args.json_path)
    _, val_df, test_df = dfs_from_metadata(metadata)
    training_df, _, _ = dfs_from_metadata(metadata)
    subset_map = {"test": test_df, "validation": val_df, "training": training_df}
    df = subset_map[args.subset]

    # Precompute baseline value per channel: transform(1 DN)[c] = -log_mean_c / log_std_c
    baseline_per_channel = [
        (-transform.log_means[c, 0, 0] / transform.log_stds[c, 0, 0]).item()
        for c in range(N_CHANNELS)
    ]

    # -- Pick balanced samples -----------------------------------------------
    print(f"Picking up to {args.n_samples} samples from '{args.subset}' set...")
    samples = pick_balanced_samples(df, args.n_samples)
    if not samples:
        raise RuntimeError("No valid samples found.")
    print(f"  Got {len(samples)} samples "
          f"({sum(1 for _, l in samples if l==1)} flare, "
          f"{sum(1 for _, l in samples if l==0)} non-flare)")

    # -- Stability loop ------------------------------------------------------
    print(f"\nRunning {args.n_runs} KernelSHAP runs per sample "
          f"(n_shap={args.n_shap})...")
    all_results = []
    for s_idx, (img, lbl) in enumerate(samples):
        class_name = "Flare" if lbl == 1 else "Non-flare"
        print(f"  Sample {s_idx + 1}/{len(samples)} ({class_name}):")
        runs = []
        for run_idx in range(args.n_runs):
            seed = run_idx * 1000 + s_idx
            importances = compute_shap_one_run(
                img, transform, baseline_per_channel,
                model, device, args.n_shap, seed, resize_to=resize_to,
            )
            runs.append(importances)
            print(f"    run {run_idx}: [{', '.join(f'{v:.4f}' for v in importances)}]")
        runs_matrix = np.array(runs)   # [n_runs, 7]
        _, mean_tau = all_pairs_kendall(runs_matrix)
        mean_cv = cv_per_channel(runs_matrix).mean()
        print(f"    → mean Kendall's τ = {mean_tau:.3f}, mean CV = {mean_cv:.3f}")
        all_results.append({"label": lbl, "runs": runs_matrix})

    # -- Summary -------------------------------------------------------------
    all_taus = [all_pairs_kendall(r["runs"])[1] for r in all_results]
    all_cvs  = [cv_per_channel(r["runs"]).mean() for r in all_results]
    print(f"\n--- Stability Summary ---")
    print(f"  Mean Kendall's τ across samples: {np.mean(all_taus):.3f} ± {np.std(all_taus):.3f}")
    print(f"  Mean CV across samples:          {np.mean(all_cvs):.3f} ± {np.std(all_cvs):.3f}")
    if np.mean(all_taus) > 0.7:
        print("  ✅ Rankings are stable (τ > 0.7)")
    elif np.mean(all_taus) > 0.4:
        print("  ⚠️  Rankings are moderately stable (0.4 < τ ≤ 0.7)")
    else:
        print("  ❌ Rankings are unstable (τ ≤ 0.4) — increase n_shap")

    # -- Figures -------------------------------------------------------------
    print("\nGenerating figures...")
    plot_variability(all_results, args.output_dir)
    plot_kendall_heatmap(all_results, args.n_runs, args.output_dir)
    plot_cv(all_results, args.output_dir)
    print(f"\nOutputs written to: {args.output_dir}/")


if __name__ == "__main__":
    main()
