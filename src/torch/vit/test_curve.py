"""Test-set metric-vs-epoch curve, computed post-hoc from saved epoch checkpoints.

Motivation
----------
train.py already logs a train/val loss-and-metric curve to wandb every epoch, live, as training
proceeds. This script computes the equivalent curve for the *test* set -- but deliberately NOT
inside the training loop and NOT live. Test data should stay a look-but-don't-touch check: this
script is meant to be run once, after training has already finished and a checkpoint strategy has
already been decided from the val curve, purely to visualize the generalization gap. It should
not become another way to pick which epoch to ship -- that's what val is for.

Requires the run to have been trained with --save-all-epochs (checkpoints under
outputs/<run-name>/epoch_checkpoints/epoch_NN.pth).

No AARP aggregation (max/mean pooling per active region) is computed anywhere in this script --
raw image-level (per-frame) metrics only, per project convention. Do not add aggregation here
unless explicitly asked.

Epoch numbering: checkpoint filenames (epoch_NN.pth) are 0-indexed; wandb's train/epoch field is
1-indexed (epoch+1). Both are recorded here (epoch_file / epoch_wandb) to avoid the mixup
documented in feedback_epoch_indexing_convention -- when pushing to wandb, test/epoch uses the
1-indexed convention so it lines up with the existing train/epoch axis.

Usage
-----
    # ViT-pretrained, push test/* metrics back into the original wandb run
    python -m src.torch.vit.test_curve \\
        --run-name pretrained-vit-solar-dataset-dropout03 \\
        --model-type vit-pretrained \\
        --json-path solar_dataset.json --stats-file stats.pkl

    # local CSV/PNG only, skip wandb entirely
    python -m src.torch.vit.test_curve \\
        --run-name cv-fold0-pretrained-vit-aarp-sampler \\
        --model-type vit-pretrained \\
        --json-path cv_folds/fold_0.json --stats-file cv_folds/fold_0_stats.pkl \\
        --no-push-wandb

In the wandb UI, set the panel's x-axis to "test/epoch" (matches "train/epoch") to overlay
against the existing val curve for the same run.
"""

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import wandb
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from tqdm import tqdm

from aarp_ml.torch.dataset import aia_euv
from aarp_ml.torch.model import DeepFlare_ViT, build_pretrained_vit
from src.torch.vit.evaluate_aarp import (
    CONFUSION_MATRIX_CLASSES,
    DEFAULT_THRESHOLD,
    _load_transform,
    compute_skill_scores,
)

N_CHANNELS = 7
N_CLASSES = 2
DEFAULT_BATCH_SIZE = 32
WANDB_PROJECT = "flare_torch"


# ---------------------------------------------------------------------------
# Checkpoint discovery
# ---------------------------------------------------------------------------

def find_epoch_checkpoints(run_name: str) -> List[Tuple[int, Path]]:
    """Return (epoch_file, path) pairs for every epoch_NN.pth, sorted by epoch.

    epoch_file is 0-indexed, matching the checkpoint filename convention in train.py.
    """
    ckpt_dir = Path("outputs") / run_name / "epoch_checkpoints"
    if not ckpt_dir.is_dir():
        raise FileNotFoundError(
            f"{ckpt_dir} not found -- this run needs to have been trained with "
            "--save-all-epochs for a per-epoch test curve to be possible."
        )
    pattern = re.compile(r"epoch_(\d+)\.pth$")
    checkpoints = []
    for p in ckpt_dir.glob("epoch_*.pth"):
        m = pattern.search(p.name)
        if m:
            checkpoints.append((int(m.group(1)), p))
    if not checkpoints:
        raise FileNotFoundError(f"No epoch_NN.pth files found under {ckpt_dir}")
    checkpoints.sort(key=lambda x: x[0])
    return checkpoints


# ---------------------------------------------------------------------------
# Model + inference
# ---------------------------------------------------------------------------

def build_model_arch(model_type: str, stats_file: str):
    """Construct the (untrained) architecture once. Weights get swapped in per checkpoint so the
    (potentially large, e.g. 304M-param pretrained vit_l_16) architecture isn't rebuilt from
    scratch for every epoch."""
    transform = _load_transform(stats_file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_type == "vit":
        model = DeepFlare_ViT(height=512, n_classes=N_CLASSES, n_passbands=N_CHANNELS).model
        resize_to = None
    elif model_type == "vit-pretrained":
        model = build_pretrained_vit(n_channels=N_CHANNELS, n_classes=N_CLASSES, pretrained=False)
        resize_to = 224
    else:
        raise ValueError(
            f"Unsupported model_type '{model_type}' -- test_curve.py covers vit/vit-pretrained "
            "only (xgb has no epoch checkpoints to curve over)."
        )

    model.to(device)
    return model, transform, device, resize_to


def evaluate_on_test(
    model: torch.nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    resize_to: int,
    threshold: float,
) -> Dict[str, float]:
    """Raw image-level (no AARP aggregation) loss + skill scores for the model's current weights."""
    resize = v2.Resize(resize_to) if resize_to is not None else None
    criterion = torch.nn.CrossEntropyLoss()

    model.eval()
    total_loss = 0.0
    y_true, y_prob = [], []
    with torch.inference_mode():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            if resize is not None:
                images = resize(images)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * labels.size(0)
            probs = F.softmax(outputs, dim=1)[:, 1]
            y_true.extend(labels.cpu().numpy())
            y_prob.extend(probs.cpu().numpy())

    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=CONFUSION_MATRIX_CLASSES)
    scores = compute_skill_scores(cm, y_true, y_prob, y_pred)
    scores["test_loss"] = total_loss / len(test_loader.dataset)
    return scores


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def save_curve_plot(df: pd.DataFrame, png_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(df["epoch_wandb"], df["test_loss"], marker="o", lw=1.2)
    axes[0].set_xlabel("Epoch (wandb, 1-indexed)")
    axes[0].set_ylabel("Test loss")
    axes[0].set_title("Test loss vs epoch")

    axes[1].plot(df["epoch_wandb"], df["precision"], marker="o", lw=1.2, label="precision")
    axes[1].plot(df["epoch_wandb"], df["recall"], marker="o", lw=1.2, label="recall")
    axes[1].plot(df["epoch_wandb"], df["TSS"], marker="o", lw=1.2, label="TSS")
    axes[1].axhline(df["prevalence"].iloc[0], color="k", ls="--", lw=0.8,
                    label=f"prevalence = {df['prevalence'].iloc[0]:.2f}")
    axes[1].set_xlabel("Epoch (wandb, 1-indexed)")
    axes[1].set_title("Test image-level skill scores vs epoch\n(no AARP aggregation)")
    axes[1].legend(loc="best", fontsize=8)

    plt.tight_layout()
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# wandb lookup
# ---------------------------------------------------------------------------

def find_wandb_run(run_name: str, entity: str = None):
    """Locate the wandb run matching run_name so test/* metrics can be appended to it."""
    api = wandb.Api()
    project_path = f"{entity}/{WANDB_PROJECT}" if entity else WANDB_PROJECT
    runs = list(api.runs(project_path, filters={"display_name": run_name}))
    if not runs:
        raise RuntimeError(
            f"No wandb run named '{run_name}' found in project '{project_path}'. "
            "Pass --wandb-entity if this project isn't under your default entity, "
            "or use --no-push-wandb to skip pushing and just get the local CSV/PNG."
        )
    if len(runs) > 1:
        print(f"  Warning: {len(runs)} wandb runs named '{run_name}' -- using the most recent.")
        runs.sort(key=lambda r: r.created_at, reverse=True)
    return runs[0]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test-set metric-vs-epoch curve, computed post-hoc from saved epoch checkpoints.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--run-name", required=True,
                        help="Run name under outputs/<run-name>/epoch_checkpoints/ "
                             "(also used to find the matching wandb run to push to).")
    parser.add_argument("--model-type", default="vit", choices=["vit", "vit-pretrained"])
    parser.add_argument("--json-path", default="solar_dataset.json")
    parser.add_argument("--stats-file", default="stats.pkl")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--output-dir", default=None,
                        help="Defaults to outputs/<run-name>/test_curve/")
    parser.add_argument("--no-push-wandb", action="store_true",
                        help="Skip pushing points back to the original wandb run; "
                             "write the local CSV/PNG only.")
    parser.add_argument("--wandb-entity", default=None,
                        help="Only needed if the run isn't under your default wandb entity.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir or Path("outputs") / args.run_name / "test_curve")
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoints = find_epoch_checkpoints(args.run_name)
    print(f"Found {len(checkpoints)} epoch checkpoints for run '{args.run_name}' "
          f"(epoch_{checkpoints[0][0]:02d} .. epoch_{checkpoints[-1][0]:02d}, filename/0-indexed).")

    model, transform, device, resize_to = build_model_arch(args.model_type, args.stats_file)
    print(f"Architecture : {args.model_type}  |  device: {device}")

    test_dataset = aia_euv(args.json_path, subset="test", transform=v2.Compose([transform]))
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    n_aarp = len(set(item["aarp_id"] for item in test_dataset.data))
    print(f"Test set     : {len(test_dataset)} frames, {n_aarp} AARPs "
          f"(image-level only -- no AARP aggregation computed here)")

    csv_path = output_dir / "test_curve.csv"
    png_path = output_dir / "test_curve.png"

    rows = []
    for epoch_file, ckpt_path in tqdm(checkpoints, desc="Evaluating checkpoints"):
        # Load to CPU first -- map_location=device would pull the whole checkpoint (including
        # the Adam optimizer state, ~2x model size) onto the GPU, and nothing here needs the
        # optimizer state at all. Explicit cleanup after so the next checkpoint's load doesn't
        # accumulate on top of this one's freed-but-cached CPU/GPU memory.
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        del ckpt
        if device.type == "cuda":
            torch.cuda.empty_cache()
        scores = evaluate_on_test(model, test_loader, device, resize_to, args.threshold)
        rows.append({
            "epoch_file": epoch_file,        # 0-indexed, matches epoch_NN.pth filenames
            "epoch_wandb": epoch_file + 1,   # 1-indexed, matches wandb's train/epoch field
            **scores,
        })

        # Save after every checkpoint, not just at the end -- evaluating dozens of epochs can
        # take a long time (especially on a contended machine), so partial results stay
        # recoverable if the run gets interrupted.
        df = pd.DataFrame(rows).sort_values("epoch_file").reset_index(drop=True)
        df.to_csv(csv_path, index=False)
        save_curve_plot(df, png_path)

    print(f"\n✓ Test curve CSV  → {csv_path}")
    print(f"✓ Test curve plot → {png_path}")

    if not args.no_push_wandb:
        print(f"\nLooking up wandb run '{args.run_name}'...")
        run_info = find_wandb_run(args.run_name, entity=args.wandb_entity)
        print(f"  Found run id={run_info.id} (entity={run_info.entity}, "
              f"project={run_info.project}) -- resuming to append test/* metrics.")
        run = wandb.init(id=run_info.id, entity=run_info.entity, project=run_info.project,
                         resume="must")
        for _, row in df.iterrows():
            run.log({
                "test/epoch": row["epoch_wandb"],
                "test/loss": row["test_loss"],
                "test/precision": row["precision"],
                "test/recall": row["recall"],
                "test/TSS": row["TSS"],
                "test/HSS": row["HSS"],
                "test/prevalence": row["prevalence"],
            })
        run.finish()
        print("  ✓ Pushed test/* metrics to wandb. In the UI, set this panel's x-axis to "
              "'test/epoch' (matches 'train/epoch') to overlay against the val curve.")
    else:
        print("\n--no-push-wandb given: skipping wandb push, CSV/PNG only.")


if __name__ == "__main__":
    main()
