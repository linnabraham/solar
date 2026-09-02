"""Model Randomization Test for Saliency Maps (Adebayo et al., NeurIPS 2018).

Sanity check for Integrated Gradients: progressively reinitializes model
weights from the output layer (top) inward and recomputes IG attributions on
the same fixed inputs.  A well-behaved attribution method must be sensitive
to model parameters — maps should diverge from the trained baseline as more
layers are randomized.

Cascade order (top → bottom):
  1. mlp_head          (final Linear classifier)
  2. transformer_norm  (LayerNorm before head)
  3. transformer_block_3 … transformer_block_0  (4 ViT blocks, last → first)
  4. patch_embedding   (input patch projection)

Outputs written to --output-dir:
  attribution_cascade_grid.png   — attribution maps across cascade levels
  ssim_vs_cascade.png            — SSIM vs randomization depth
  spearman_vs_cascade.png        — Spearman ρ vs randomization depth

Usage:
    python -m src.torch.vit.model_randomization_test \\
        --output-dir plots/model_randomization_test \\
        --n-samples 2 \\
        --n-steps 25 \\
        --channel 1 \\
        --subset test
"""

import argparse
import copy
import gc
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from captum.attr import IntegratedGradients
from scipy.stats import spearmanr
from skimage.metrics import structural_similarity

from torchvision.transforms import v2

from aarp_ml.dataset import all_wavelengths
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

CHANNEL_NAMES = ["94 Å", "131 Å", "171 Å", "193 Å", "211 Å", "304 Å", "335 Å"]
VALID_MODEL_TYPES = {"vit": "deepflare_vit", "vit-pretrained": "vit_pretrained"}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Model Randomization Sanity Check for IG Saliency Maps"
    )
    parser.add_argument("--model-path", default=TRAINED_MODEL_PATH,
                        help="Path to trained ViT model checkpoint.")
    parser.add_argument("--output-dir", default=None,
                        help="Directory for output figures. Defaults to "
                             "plots/model_randomization_test/<run-id> derived from --model-path.")
    parser.add_argument("--n-samples", type=int, default=2,
                        help="Number of samples to use (tries 1 flare + 1 non-flare)")
    parser.add_argument("--n-steps", type=int, default=25,
                        help="IG integration steps (default 25 for speed; paper uses 50-300)")
    parser.add_argument("--channel", type=int, default=1,
                        help="AIA channel index to display in the grid figure (0-6)")
    parser.add_argument("--subset", default="test",
                        choices=["test", "validation", "training"])
    parser.add_argument("--json-path", default="solar_dataset.json")
    parser.add_argument("--stats-file", default="stats.pkl")
    parser.add_argument("--model-type", default="vit", choices=list(VALID_MODEL_TYPES),
                        help="Model architecture: 'vit' (DeepFlare_ViT, default) or "
                             "'vit-pretrained' (torchvision vit_l_16). The pretrained "
                             "architecture has 24 transformer blocks vs. 4, so its "
                             "cascade has 27 levels instead of 7 -- proportionally more "
                             "IG calls per run.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def pick_one_sample(df, label):
    """Return (image_np [7,512,512], int_label) for the middle timestep of the
    first AARP in *df* that has *label*.  Returns (None, None) if not found."""
    matching = df[df["label"] == label]["aarp_id"].unique()
    if len(matching) == 0:
        return None, None
    aarp_id = int(matching[0])
    aarp_df = df[df["aarp_id"] == aarp_id]
    s = single_aarp(aarp_id, aarp_df)
    images = s.get_images()          # [T, 7, 512, 512]
    mid = len(images) // 2
    return images[mid], int(s.label) # [7, 512, 512]


# ---------------------------------------------------------------------------
# IG with configurable n_steps
# ---------------------------------------------------------------------------

def compute_ig(model, image_np, label, transform, device, n_steps, resize_to=None):
    """Compute IG attribution for a single image (numpy [7,512,512]).

    Returns numpy array [7, H, W] -- H,W is 512 (native), or resize_to when given.
    Reference and randomized-cascade attributions are always computed through
    this same function with the same resize_to, so they stay directly comparable
    to each other regardless of the model's native input size.
    """
    tensor_img = torch.from_numpy(image_np).to(torch.float32)
    tensor_data = transform(tensor_img)
    if resize_to is not None:
        tensor_data = v2.Resize(resize_to)(tensor_data)
    tensor_data = tensor_data.to(device)
    baseline = transform(torch.zeros_like(tensor_img))
    if resize_to is not None:
        baseline = v2.Resize(resize_to)(baseline)
    baseline = baseline.to(device)

    model.eval()
    ig = IntegratedGradients(model, multiply_by_inputs=True)
    target = torch.tensor(label, dtype=torch.int32)

    attrs, _ = ig.attribute(
        tensor_data.unsqueeze(0),
        baseline.unsqueeze(0),
        target=target,
        n_steps=n_steps,
        internal_batch_size=1,
        return_convergence_delta=True,
    )
    result = attrs[0].detach().cpu().numpy()   # [7, 512, 512]
    del tensor_img, tensor_data, baseline, attrs
    gc.collect()
    torch.cuda.empty_cache()
    return result


# ---------------------------------------------------------------------------
# Layer randomization
# ---------------------------------------------------------------------------

def get_cascade_groups(model, model_type="deepflare_vit"):
    """Return layer groups in cascading order: top (output) → bottom (input).

    Each entry is (name_str, list_of_nn_Modules_or_Parameters).

    model_type == "vit_pretrained" (torchvision vit_l_16) has an entirely
    different internal module tree than vit_pytorch's ViT (DeepFlare_ViT), and
    24 transformer blocks instead of 4 -- so this is a parallel cascade
    definition, not a reuse of the same attribute names. model.class_token is
    a raw nn.Parameter (no reset_parameters()) that torchvision itself
    initializes to zeros -- grouped into the bottom "patch_embedding" step
    alongside conv_proj, since (unlike model.to_latent/model.dropout below,
    which have no learnable parameters at all) it does carry real trained
    weights and skipping it would leave one parameter permanently untouched
    through the whole cascade.
    """
    if model_type == "vit_pretrained":
        groups = []
        groups.append(("mlp_head",         [model.heads.head]))
        groups.append(("transformer_norm", [model.encoder.ln]))
        n_blocks = len(model.encoder.layers)
        for i in reversed(range(n_blocks)):
            groups.append((f"transformer_block_{i}", [model.encoder.layers[i]]))
        groups.append(("patch_embedding",  [model.conv_proj, model.class_token]))
        return groups

    # DeepFlare_ViT (vit_pytorch's ViT). model.to_latent (Identity) and
    # model.dropout (Dropout) have no trainable parameters and are
    # intentionally skipped.
    groups = []
    groups.append(("mlp_head",         [model.mlp_head]))
    groups.append(("transformer_norm", [model.transformer.norm]))
    n_blocks = len(model.transformer.layers)
    for i in reversed(range(n_blocks)):
        groups.append((f"transformer_block_{i}", [model.transformer.layers[i]]))
    groups.append(("patch_embedding",  [model.to_patch_embedding]))
    return groups


def randomize_modules(modules):
    """Re-initialize all learnable layers/parameters inside *modules* in-place.

    Uses each layer's own reset_parameters() where available (same distribution
    as original initialization — Kaiming uniform for Linear, etc.), falling back
    to Xavier uniform for layers that lack reset_parameters(). A raw
    nn.Parameter (e.g. vit_l_16's class_token, which has no reset_parameters()
    of its own) is reset to zeros, matching torchvision's own init for it.
    """
    for mod in modules:
        if isinstance(mod, nn.Parameter):
            nn.init.zeros_(mod)
            continue
        for layer in mod.modules():
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()
            elif hasattr(layer, "weight") and layer.weight is not None:
                nn.init.xavier_uniform_(layer.weight)
                if hasattr(layer, "bias") and layer.bias is not None:
                    nn.init.zeros_(layer.bias)


# ---------------------------------------------------------------------------
# Comparison metrics
# ---------------------------------------------------------------------------

def compare_attributions(ref, rand):
    """Return mean SSIM and mean Spearman ρ between two [C,H,W] attribution maps."""
    ssim_vals, corr_vals = [], []
    for c in range(ref.shape[0]):
        r, t = ref[c], rand[c]
        data_range = float(max(r.max() - r.min(), t.max() - t.min(), 1e-8))
        ssim_vals.append(structural_similarity(r, t, data_range=data_range))
        corr_vals.append(float(spearmanr(r.ravel(), t.ravel()).statistic))
    return float(np.mean(ssim_vals)), float(np.mean(corr_vals))


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_cascade_grid(results, samples, channel_idx, output_dir):
    """Figure A — attribution maps: rows = cascade levels, cols = samples.

    Each cell shows the attribution for *channel_idx*. Row labels include the
    cascade level name plus SSIM/ρ relative to the trained model.
    """
    n_levels = len(results)
    n_samples = len(samples)
    channel_name = CHANNEL_NAMES[channel_idx]

    fig, axes = plt.subplots(
        n_levels, n_samples,
        figsize=(4.5 * n_samples, 3.2 * n_levels),
        squeeze=False,
    )

    # Shared colour scale: 99th-percentile of absolute values across all maps
    all_maps = [r["attrs"][si][channel_idx] for r in results for si in range(n_samples)]
    vmax = float(np.percentile([np.abs(m).max() for m in all_maps], 99)) * 0.99
    vmin = -vmax

    fig.suptitle(
        f"IG Attribution Maps — Cascading Model Randomization  ({channel_name})",
        fontsize=12, y=1.01,
    )

    im_ref = None
    for row, result in enumerate(results):
        # Build left-margin label
        row_label = result["label"]
        if result["ssim"] is not None:
            row_label += f"\nSSIM={result['ssim']:.3f}  ρ={result['corr']:.3f}"

        for col, (_, lbl) in enumerate(samples):
            ax = axes[row][col]
            attr_map = result["attrs"][col][channel_idx]
            im = ax.imshow(attr_map, cmap="seismic", vmin=vmin, vmax=vmax)
            im_ref = im
            ax.axis("off")

            if col == 0:
                ax.annotate(
                    row_label,
                    xy=(0, 0.5), xycoords="axes fraction",
                    xytext=(-10, 0), textcoords="offset points",
                    fontsize=7, ha="right", va="center",
                    annotation_clip=False,
                )
            if row == 0:
                ax.set_title(
                    f"{'Flare' if lbl == 1 else 'Non-flare'}",
                    fontsize=9,
                )

    if im_ref is not None:
        fig.colorbar(im_ref, ax=axes, orientation="vertical",
                     fraction=0.02, pad=0.04, label="IG Attribution")

    out_path = os.path.join(output_dir, "attribution_cascade_grid.png")
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")
    return out_path


def plot_metric_curve(x_labels, values, ylabel, title, out_path,
                      color, chance_y=0.0, perfect_y=None):
    """Generic line plot for a single metric vs cascade depth."""
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(range(len(values)), values, "o-", color=color, linewidth=2, markersize=8)
    ax.axhline(chance_y, color="red", linestyle="--", alpha=0.6,
               label=f"Chance ({chance_y})")
    if perfect_y is not None:
        ax.axhline(perfect_y, color="green", linestyle="--", alpha=0.4,
                   label=f"Perfect ({perfect_y})")
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Cascade level  (top → bottom)")
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_metrics(results, output_dir):
    """Figures B and C: SSIM and Spearman ρ vs cascade depth."""
    labels = [r["label"] for r in results]
    ssims  = [r["ssim"] for r in results]
    corrs  = [r["corr"] for r in results]

    plot_metric_curve(
        labels, ssims,
        ylabel="SSIM  (vs trained model)",
        title="Model Randomization Test — SSIM of IG Attributions",
        out_path=os.path.join(output_dir, "ssim_vs_cascade.png"),
        color="steelblue",
        chance_y=0.0,
        perfect_y=1.0,
    )
    plot_metric_curve(
        labels, corrs,
        ylabel="Spearman ρ  (vs trained model)",
        title="Model Randomization Test — Spearman Correlation of IG Attributions",
        out_path=os.path.join(output_dir, "spearman_vs_cascade.png"),
        color="darkorange",
        chance_y=0.0,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    from pathlib import Path
    args = parse_args()
    if args.output_dir is None:
        run_id = Path(args.model_path).parent.name
        args.output_dir = f"plots/model_randomization_test/{run_id}"
    os.makedirs(args.output_dir, exist_ok=True)

    # -- Load model and data --------------------------------------------------
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
    training_df, val_df, test_df = dfs_from_metadata(metadata)
    subset_map = {"test": test_df, "validation": val_df, "training": training_df}
    df = subset_map[args.subset]

    # -- Pick one flare + one non-flare sample --------------------------------
    print(f"Picking up to {args.n_samples} sample(s) from '{args.subset}' set...")
    samples = []
    for lbl in [1, 0]:
        if len(samples) >= args.n_samples:
            break
        img, int_lbl = pick_one_sample(df, lbl)
        if img is not None:
            samples.append((img, int_lbl))
            print(f"  {'Flare' if int_lbl == 1 else 'Non-flare'} sample, shape={img.shape}")

    if not samples:
        raise RuntimeError(f"No samples found in '{args.subset}' subset.")

    # -- Reference attributions (trained model) --------------------------------
    print(f"\nComputing reference IG attributions (n_steps={args.n_steps})...")
    ref_attrs = [
        compute_ig(model, img, lbl, transform, device, n_steps=args.n_steps, resize_to=resize_to)
        for img, lbl in samples
    ]
    results = [{"label": "trained", "attrs": ref_attrs, "ssim": 1.0, "corr": 1.0}]

    # -- Cascading randomization ----------------------------------------------
    randomized_model = copy.deepcopy(model)
    cascade_groups = get_cascade_groups(randomized_model, model_type=config.model_type)

    print("\nRunning cascade randomization:")
    for name, modules in cascade_groups:
        print(f"  Randomizing: {name} ...", end=" ", flush=True)
        randomize_modules(modules)          # mutate in-place (cascading)

        rand_attrs = [
            compute_ig(randomized_model, img, lbl, transform, device, n_steps=args.n_steps,
                      resize_to=resize_to)
            for img, lbl in samples
        ]

        # Average SSIM and ρ over all samples
        per_sample_ssim, per_sample_corr = [], []
        for ref, rand in zip(ref_attrs, rand_attrs):
            s, c = compare_attributions(ref, rand)
            per_sample_ssim.append(s)
            per_sample_corr.append(c)

        mean_ssim = float(np.mean(per_sample_ssim))
        mean_corr = float(np.mean(per_sample_corr))
        print(f"SSIM={mean_ssim:.4f}  ρ={mean_corr:.4f}")

        results.append({
            "label": name,
            "attrs": rand_attrs,
            "ssim": mean_ssim,
            "corr": mean_corr,
        })

    # -- Generate figures ------------------------------------------------------
    print("\nGenerating figures...")
    plot_cascade_grid(results, samples, channel_idx=args.channel,
                      output_dir=args.output_dir)
    plot_metrics(results, output_dir=args.output_dir)

    # -- Summary table ---------------------------------------------------------
    print("\n--- Summary ---")
    print(f"{'Level':<25}  {'SSIM':>8}  {'Spearman ρ':>12}")
    print("-" * 50)
    for r in results:
        ssim_str = f"{r['ssim']:.4f}" if r["ssim"] is not None else "  —"
        corr_str = f"{r['corr']:.4f}" if r["corr"] is not None else "  —"
        print(f"{r['label']:<25}  {ssim_str:>8}  {corr_str:>12}")

    print(f"\nOutputs written to: {args.output_dir}/")


if __name__ == "__main__":
    main()
