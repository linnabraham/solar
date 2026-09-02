"""SHAP Faithfulness Test — channel deletion curve.

Validates that KernelSHAP correctly identifies which AIA passbands the model
relies on. Channels are sorted by mean |SHAP| importance and progressively
zeroed out to the baseline value. If SHAP is faithful, masking the most
important channels first should cause the steepest performance drop.

Three orderings are compared:
  - SHAP order      (most → least important by mean |SHAP|)
  - Reverse-SHAP    (least → most — should drop slowest)
  - Random baseline (10 shuffled orderings — shaded band)

Metrics: Precision and Recall (from existing compute_metrics() in test.py).
No new metrics are introduced.

Reads the existing shap_stats_test.json produced by kshap.py — does NOT
overwrite or modify it.

Outputs written to --output-dir:
  faithfulness_degradation.png    — precision + recall vs channels masked
  channel_order_comparison.png    — TSS drop at k=1 across orderings

Usage:
    python -m src.torch.vit.shap_faithfulness_test \\
        --output-dir plots/shap_faithfulness_test \\
        --shap-json  shap_stats_test.json \\
        --subset test
"""

import argparse
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from aarp_ml.torch.dataset import aia_euv
from src.torch.vit.analyze_shap import (
    AIA_WAVELENGTHS,
    SHAPAnalysisConfig,
    channel_columns,
    load_and_process,
)
from src.torch.vit.test import compute_metrics
from src.torch.vit.train import TrainingConfig
from src.torch.vit.utils import get_model_and_transform, needs_resize

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TRAINED_MODEL_PATH = "outputs/glad-shape-197/trained_model.pth"
N_CHANNELS = 7
CONFUSION_MATRIX_CLASSES = [0, 1]
VALID_MODEL_TYPES = {"vit": "deepflare_vit", "vit-pretrained": "vit_pretrained"}

# Map 'AIA_94' → channel index 0, 'AIA_131' → 1, …
CHANNEL_NAME_TO_IDX = {f"AIA_{w}": i for i, w in enumerate(AIA_WAVELENGTHS)}
CHANNEL_IDX_TO_LABEL = {i: f"{w} Å" for i, w in enumerate(AIA_WAVELENGTHS)}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="SHAP Faithfulness Test — channel deletion curve"
    )
    parser.add_argument("--model-path", default=TRAINED_MODEL_PATH,
                        help="Path to trained ViT model checkpoint.")
    parser.add_argument("--output-dir", default=None,
                        help="Directory for output plots. Defaults to "
                             "plots/shap_faithfulness_test/<run-id> derived from --model-path.")
    parser.add_argument("--shap-json", default="shap_stats_test.json",
                        help="Path to existing shap_stats JSON (read-only)")
    parser.add_argument("--n-random", type=int, default=10,
                        help="Number of random channel orderings for baseline band")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--subset", default="test",
                        choices=["test", "validation", "training"])
    parser.add_argument("--json-path", default="solar_dataset.json")
    parser.add_argument("--stats-file", default="stats.pkl")
    parser.add_argument("--model-type", default="vit", choices=list(VALID_MODEL_TYPES),
                        help="Model architecture: 'vit' (DeepFlare_ViT, default) or "
                             "'vit-pretrained' (torchvision vit_l_16).")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Load SHAP ordering (read-only from existing JSON)
# ---------------------------------------------------------------------------

def get_shap_channel_order(shap_json_path):
    """Load shap_stats JSON and return channels sorted by mean |SHAP| (most → least).

    Returns:
        ordered_names  — list of channel names e.g. ['AIA_94', 'AIA_211', ...]
        ordered_labels — human-readable e.g. ['94 Å', '211 Å', ...]
        mean_abs       — dict {channel_name: mean_abs_importance}
    """
    cfg = SHAPAnalysisConfig(input_json=shap_json_path, correct_only=False)
    df = load_and_process(cfg)
    channels = channel_columns()   # canonical order: AIA_94, AIA_131, …, AIA_335
    mean_abs = df[channels].abs().mean()
    ordered = mean_abs.sort_values(ascending=False)
    ordered_names = ordered.index.tolist()
    ordered_labels = [f"{AIA_WAVELENGTHS[CHANNEL_NAME_TO_IDX[n]]} Å" for n in ordered_names]
    return ordered_names, ordered_labels, mean_abs.to_dict()


# ---------------------------------------------------------------------------
# Inference with masked channels
# ---------------------------------------------------------------------------

def evaluate_masked(model, loader, masked_channel_indices, baseline_per_channel, device,
                    resize_to=None):
    """Run inference with specified channels replaced by their baseline value.

    Args:
        masked_channel_indices: list of int channel indices to zero-out.
        baseline_per_channel:   list[float] — one scalar per channel.
        resize_to: if given (e.g. 224 for the pretrained vit_l_16), resize after
            masking (masking fills a channel with a uniform scalar, so resize
            order doesn't affect the result -- a constant channel stays constant
            under bilinear interpolation).

    Returns:
        dict with 'precision', 'recall', 'TP', 'FP', 'FN', 'TN'.
    """
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for features, labels in loader:
            features = features.clone().to(device)
            for c in masked_channel_indices:
                features[:, c, :, :] = baseline_per_channel[c]
            if resize_to is not None:
                features = v2.Resize(resize_to)(features)
            outputs = model(features)
            preds = outputs.argmax(dim=1)
            y_true.extend(labels.tolist())
            y_pred.extend(preds.cpu().tolist())
    cm = confusion_matrix(y_true, y_pred, labels=CONFUSION_MATRIX_CLASSES)
    return compute_metrics(cm)   # precision, recall, TP, FP, FN, TN


def run_deletion_curve(model, loader, ordering_names, baseline_per_channel, device,
                       resize_to=None):
    """Sweep deletion levels 0 → 7, returning lists of precision and recall."""
    precisions, recalls = [], []
    masked_so_far = []
    # Level 0: no channels masked
    m0 = evaluate_masked(model, loader, [], baseline_per_channel, device, resize_to=resize_to)
    precisions.append(m0["precision"])
    recalls.append(m0["recall"])
    for ch_name in ordering_names:
        masked_so_far.append(CHANNEL_NAME_TO_IDX[ch_name])
        m = evaluate_masked(model, loader, masked_so_far, baseline_per_channel, device,
                            resize_to=resize_to)
        precisions.append(m["precision"])
        recalls.append(m["recall"])
    return np.array(precisions), np.array(recalls)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _x_tick_labels(ordering_names):
    """Build x-axis tick labels: '0\nnone', '1\n94Å', '2\n+211Å', …"""
    labels = ["0\n(none)"]
    for i, name in enumerate(ordering_names, start=1):
        w = AIA_WAVELENGTHS[CHANNEL_NAME_TO_IDX[name]]
        prefix = "+" if i > 1 else ""
        labels.append(f"{i}\n{prefix}{w}Å")
    return labels


def plot_faithfulness(
    shap_prec, shap_rec,
    rev_prec, rev_rec,
    rand_prec_stack, rand_rec_stack,
    shap_ordering_names,
    output_dir,
):
    """Figure A: precision + recall degradation curves."""
    x = np.arange(N_CHANNELS + 1)
    tick_labels = _x_tick_labels(shap_ordering_names)

    # Random band statistics
    rand_prec_mean = rand_prec_stack.mean(axis=0)
    rand_prec_std  = rand_prec_stack.std(axis=0)
    rand_rec_mean  = rand_rec_stack.mean(axis=0)
    rand_rec_std   = rand_rec_stack.std(axis=0)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    for ax, metric_name, shap_vals, rev_vals, rand_mean, rand_std in [
        (ax1, "Recall",    shap_rec,  rev_rec,  rand_rec_mean,  rand_rec_std),
        (ax2, "Precision", shap_prec, rev_prec, rand_prec_mean, rand_prec_std),
    ]:
        # SHAP ordering
        ax.plot(x, shap_vals, "o-", color="steelblue", linewidth=2.5,
                markersize=7, label="SHAP order (most → least)", zorder=4)
        # Reverse SHAP
        ax.plot(x, rev_vals, "s--", color="darkorange", linewidth=1.8,
                markersize=6, label="Reverse-SHAP (least → most)", zorder=3)
        # Random band
        ax.fill_between(x, rand_mean - rand_std, rand_mean + rand_std,
                        alpha=0.25, color="grey", label="Random ± 1 std")
        ax.plot(x, rand_mean, "-", color="grey", linewidth=1.2,
                alpha=0.7, label="Random mean")

        ax.axhline(0, color="red", linestyle=":", alpha=0.5, linewidth=0.8)
        ax.set_ylabel(metric_name, fontsize=11)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3)

    ax2.set_xticks(x)
    ax2.set_xticklabels(tick_labels, fontsize=8)
    ax2.set_xlabel("Channels masked (cumulative, top → bottom = most → least SHAP important)",
                   fontsize=9)
    fig.suptitle("SHAP Faithfulness Test — Channel Deletion Degradation", fontsize=12)
    plt.tight_layout()
    out = os.path.join(output_dir, "faithfulness_degradation.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


def plot_single_mask_comparison(
    shap_prec, shap_rec,
    rev_prec, rev_rec,
    rand_prec_stack, rand_rec_stack,
    shap_ordering_names,
    output_dir,
):
    """Figure B: recall & precision at k=1 (one channel masked) across orderings."""
    # k=1 values
    shap_rec_k1   = shap_rec[1]
    shap_prec_k1  = shap_prec[1]
    rev_rec_k1    = rev_rec[1]
    rev_prec_k1   = rev_prec[1]
    rand_rec_k1   = rand_rec_stack[:, 1]
    rand_prec_k1  = rand_prec_stack[:, 1]

    top_ch = AIA_WAVELENGTHS[CHANNEL_NAME_TO_IDX[shap_ordering_names[0]]]
    bot_ch = AIA_WAVELENGTHS[CHANNEL_NAME_TO_IDX[shap_ordering_names[-1]]]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, metric_name, shap_val, rev_val, rand_vals, top_ch_label, bot_ch_label in [
        (axes[0], "Recall",    shap_rec_k1,  rev_rec_k1,  rand_rec_k1,
         f"SHAP top\n({top_ch} Å)", f"SHAP bottom\n({bot_ch} Å)"),
        (axes[1], "Precision", shap_prec_k1, rev_prec_k1, rand_prec_k1,
         f"SHAP top\n({top_ch} Å)", f"SHAP bottom\n({bot_ch} Å)"),
    ]:
        bar_heights = [shap_val, rev_val, rand_vals.mean()]
        bar_errs    = [0, 0, rand_vals.std()]
        bar_labels  = [top_ch_label, bot_ch_label, "Random\nmean"]
        colours     = ["steelblue", "darkorange", "grey"]

        bars = ax.bar(range(3), bar_heights, yerr=bar_errs, capsize=5,
                      color=colours, alpha=0.8)
        # Overlay random individual orderings as dots
        x_jit = np.random.uniform(-0.15, 0.15, size=len(rand_vals)) + 2
        ax.scatter(x_jit, rand_vals, s=20, color="black", alpha=0.5, zorder=5)

        ax.set_xticks(range(3))
        ax.set_xticklabels(bar_labels, fontsize=9)
        ax.set_ylabel(metric_name)
        ax.set_ylim(0, 1.05)
        ax.set_title(f"{metric_name} after masking 1 channel")
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle(
        "SHAP Faithfulness — Impact of Masking Single Most/Least Important Channel",
        fontsize=11,
    )
    plt.tight_layout()
    out = os.path.join(output_dir, "channel_order_comparison.png")
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
        args.output_dir = f"plots/shap_faithfulness_test/{run_id}"
    os.makedirs(args.output_dir, exist_ok=True)

    # -- Load SHAP ordering (read-only) --------------------------------------
    print(f"Loading SHAP stats from '{args.shap_json}' (read-only)...")
    shap_order_names, shap_order_labels, mean_abs = get_shap_channel_order(args.shap_json)
    rev_order_names = list(reversed(shap_order_names))

    print("  Global channel ordering by mean |SHAP|:")
    for rank, (name, label) in enumerate(zip(shap_order_names, shap_order_labels), 1):
        print(f"    {rank}. {label:8s}  mean|SHAP| = {mean_abs[name]:.5f}")

    # -- Load model + dataset ------------------------------------------------
    print("\nLoading model and dataset...")
    config = TrainingConfig(
        json_path=args.json_path,
        stats_file=args.stats_file,
        trained_model_path=args.model_path,
        model_type=VALID_MODEL_TYPES[args.model_type],
    )
    model, transform, device = get_model_and_transform(config)
    resize_to = needs_resize(config)
    print(f"Model type: {args.model_type}" + (f"  (resize to {resize_to})" if resize_to else ""))

    dataset = aia_euv(args.json_path, subset=args.subset, transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    print(f"  Dataset size ({args.subset}): {len(dataset)} samples")

    # Baseline value for each channel: transform(1 DN)[c] = -log_mean_c / log_std_c
    baseline_per_channel = [
        (-transform.log_means[c, 0, 0] / transform.log_stds[c, 0, 0]).item()
        for c in range(N_CHANNELS)
    ]

    # -- Deletion sweep: SHAP order ------------------------------------------
    print("\nRunning deletion sweep — SHAP order...")
    shap_prec, shap_rec = run_deletion_curve(
        model, loader, shap_order_names, baseline_per_channel, device, resize_to=resize_to
    )
    print(f"  Recall at k=0 (no mask): {shap_rec[0]:.3f}")
    print(f"  Recall at k=7 (all masked): {shap_rec[-1]:.3f}")

    # -- Deletion sweep: Reverse-SHAP ----------------------------------------
    print("\nRunning deletion sweep — Reverse-SHAP order...")
    rev_prec, rev_rec = run_deletion_curve(
        model, loader, rev_order_names, baseline_per_channel, device, resize_to=resize_to
    )

    # -- Deletion sweeps: random orderings -----------------------------------
    print(f"\nRunning {args.n_random} random orderings...")
    rand_prec_list, rand_rec_list = [], []
    base_channels = shap_order_names.copy()
    for r_idx in range(args.n_random):
        random.seed(r_idx * 17 + 3)
        rand_order = base_channels.copy()
        random.shuffle(rand_order)
        rp, rr = run_deletion_curve(model, loader, rand_order,
                                    baseline_per_channel, device, resize_to=resize_to)
        rand_prec_list.append(rp)
        rand_rec_list.append(rr)
        print(f"  [{r_idx+1}/{args.n_random}] recall@k=1: {rr[1]:.3f}")
    rand_prec_stack = np.array(rand_prec_list)   # [n_random, 8]
    rand_rec_stack  = np.array(rand_rec_list)

    # -- Summary -------------------------------------------------------------
    print("\n--- Faithfulness Summary ---")
    print(f"{'Level':<30}  {'Recall':>8}  {'Precision':>10}")
    print("-" * 53)
    for k in range(N_CHANNELS + 1):
        ch_str = "(none)" if k == 0 else f"masked top-{k}"
        print(f"{ch_str:<30}  {shap_rec[k]:>8.3f}  {shap_prec[k]:>10.3f}")

    drop_shap = shap_rec[0] - shap_rec[1]
    drop_rand = shap_rec[0] - rand_rec_stack[:, 1].mean()
    print(f"\n  Recall drop (k=0→1): SHAP={drop_shap:.3f}, Random={drop_rand:.3f}")
    if drop_shap > drop_rand:
        print("  ✅ SHAP order causes larger recall drop than random — faithfulness supported")
    else:
        print("  ⚠️  SHAP order does NOT cause larger drop than random at k=1")

    # -- Figures -------------------------------------------------------------
    print("\nGenerating figures...")
    plot_faithfulness(
        shap_prec, shap_rec,
        rev_prec, rev_rec,
        rand_prec_stack, rand_rec_stack,
        shap_order_names,
        args.output_dir,
    )
    plot_single_mask_comparison(
        shap_prec, shap_rec,
        rev_prec, rev_rec,
        rand_prec_stack, rand_rec_stack,
        shap_order_names,
        args.output_dir,
    )
    print(f"\nOutputs written to: {args.output_dir}/")


if __name__ == "__main__":
    main()
