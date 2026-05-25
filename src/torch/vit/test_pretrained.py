"""Evaluate a VIT_Pretrained (torchvision vit_l_16) checkpoint.

This is the evaluation counterpart of test.py for models trained with
VIT_Pretrained (torchvision.models.vit_l_16 fine-tuned on AIA data) rather
than the custom DeepFlare_ViT used in the main pipeline.

Key differences from test.py:
  - Loads torchvision vit_l_16 architecture (not vit-pytorch ViT)
  - Images are resized 512 → 224 at inference time (as during training)
  - Random flip transforms from VIT_Pretrained.forward() are intentionally
    skipped here — they are a training-time augmentation and should not be
    applied during evaluation.

Usage:
    python -m src.torch.vit.test_pretrained \\
        --subset test \\
        --model-path outputs/treasured-blaze-221/trained_model.pth
"""

import time
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import torch.nn as nn
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

from aarp_ml.torch.dataset import aia_euv, AIALogTransform
from src.torch.vit.test import compute_metrics        # reuse existing helper
from ml_utils.visualization import plot_confusion_matrix

# ==================== Constants ====================

DEFAULT_TRAINED_MODEL_PATH = "outputs/treasured-blaze-221/trained_model.pth"
DEFAULT_BATCH_SIZE = 32
RESIZE_SIZE = 224          # vit_l_16 expects 224×224 inputs
N_CHANNELS = 7
N_CLASSES = 2
CONFUSION_MATRIX_CLASSES = [0, 1]
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ==================== Model loading ====================

def build_pretrained_vit(n_channels: int = N_CHANNELS,
                         n_classes: int = N_CLASSES,
                         dropout_prob: float = 0.3) -> torch.nn.Module:
    """Reconstruct the torchvision vit_l_16 architecture used by VIT_Pretrained.

    Mirrors the modifications made in VIT_Pretrained.__init__:
      - conv_proj replaced to accept n_channels input channels
      - heads replaced with Dropout + Linear for n_classes outputs

    Args:
        n_channels: Number of AIA passband channels (default 7).
        n_classes: Number of output classes (default 2).
        dropout_prob: Dropout probability applied before the final linear layer.

    Returns:
        The modified torchvision VisionTransformer (the .model attribute).
    """
    model = torchvision.models.vit_l_16(weights=None)    # no pretrained weights; we load from ckpt

    conv1_out = model.conv_proj.out_channels
    model.conv_proj = nn.Conv2d(
        n_channels,
        conv1_out,
        kernel_size=(16, 16),
        stride=(16, 16),
    )

    lin_in = model.heads.head.in_features
    model.heads = nn.Sequential(
        nn.Dropout(p=dropout_prob, inplace=True),
        nn.Linear(in_features=lin_in, out_features=n_classes, bias=True),
    )
    return model


def load_pretrained_vit(checkpoint_path: str,
                        device: torch.device) -> torch.nn.Module:
    """Load a VIT_Pretrained checkpoint and return a ready-to-use model.

    Args:
        checkpoint_path: Path to the .pth checkpoint file.
        device: Torch device to map the model onto.

    Returns:
        Model in eval mode on the specified device.
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
        raise ValueError(
            f"Expected checkpoint with 'model_state_dict' key, "
            f"got top-level keys: {list(ckpt.keys()) if isinstance(ckpt, dict) else type(ckpt)}"
        )

    model = build_pretrained_vit()
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model


# ==================== Evaluation ====================

def evaluate(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    subset: str,
    output_path: str,
) -> Dict:
    """Run evaluation and save confusion matrix.

    Images are resized to RESIZE_SIZE (224) before being passed through
    the model, matching the resize applied during training.

    Args:
        model: Loaded VIT_Pretrained model in eval mode.
        data_loader: DataLoader yielding (images, labels) batches.
        device: Torch device.
        subset: Name of the split being evaluated (for display/filename).
        output_path: Where to save the confusion matrix PNG.

    Returns:
        Dict with keys TP, FP, FN, TN, precision, recall, accuracy.
    """
    resize = v2.Resize(RESIZE_SIZE)
    loss_func = nn.CrossEntropyLoss()

    val_loss = 0.0
    y_true: List[int] = []
    y_pred: List[int] = []

    with torch.inference_mode():
        for images, labels in tqdm(data_loader, desc=f"Evaluating {subset}", leave=False):
            images = resize(images.to(device))
            labels = labels.to(device)
            outputs = model(images)
            val_loss += loss_func(outputs, labels).item() * labels.size(0)
            _, predicted = torch.max(outputs, dim=1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    cm = confusion_matrix(y_true, y_pred, labels=CONFUSION_MATRIX_CLASSES)
    metrics = compute_metrics(cm)
    n_total = len(data_loader.dataset)
    accuracy = (metrics["TP"] + metrics["TN"]) / n_total
    metrics["accuracy"] = float(accuracy)

    # Print results
    print(f"\n{'='*50}")
    print(f"Confusion Matrix ({subset} set):")
    print(f"{'='*50}")
    print(cm)
    print(f"\nMetrics:")
    print(f"  TP: {metrics['TP']}  FP: {metrics['FP']}")
    print(f"  FN: {metrics['FN']}  TN: {metrics['TN']}")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  Accuracy  : {accuracy:.4f}")
    print(f"{'='*50}\n")

    # Save confusion matrix plot
    fig, ax = plot_confusion_matrix(
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
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"✓ Saved confusion matrix to {output_path}")

    return metrics


# ==================== CLI ====================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate a VIT_Pretrained (torchvision vit_l_16) checkpoint."
    )
    parser.add_argument(
        "--subset", required=True,
        choices=["training", "validation", "test"],
        help="Data split to evaluate.",
    )
    parser.add_argument(
        "--model-path", default=DEFAULT_TRAINED_MODEL_PATH,
        help="Path to trained model checkpoint (default: %(default)s).",
    )
    parser.add_argument(
        "--output-path", default=None,
        help="Output PNG for confusion matrix. "
             "Defaults to plots/cm_<run-id>_<subset>.png.",
    )
    parser.add_argument(
        "--json-path", default="solar_dataset.json",
        help="Path to dataset JSON file.",
    )
    parser.add_argument(
        "--stats-file", default="stats.pkl",
        help="Path to normalisation stats pickle.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
        help=f"Batch size (default: {DEFAULT_BATCH_SIZE}).",
    )
    args = parser.parse_args()

    run_id = Path(args.model_path).parent.name
    if args.output_path is None:
        args.output_path = f"plots/cm_{run_id}_{args.subset}.png"

    try:
        device = torch.device(DEFAULT_DEVICE)
        print(f"Device      : {device}")
        print(f"Model       : {args.model_path}")
        print(f"Subset      : {args.subset}")
        print(f"Output      : {args.output_path}")

        # Load normalisation stats
        with open(args.stats_file, "rb") as f:
            stats = pickle.load(f)
        means = [stats["mean"][f"channel_{i}"] for i in range(N_CHANNELS)]
        stds  = [stats["std"][f"channel_{i}"]  for i in range(N_CHANNELS)]
        transform = AIALogTransform(means=means, stds=stds)

        # Load dataset
        print(f"\nLoading {args.subset} dataset...")
        dataset = aia_euv(args.json_path, subset=args.subset, transform=transform)
        data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
        print(f"  {len(dataset)} samples")

        # Load model
        print(f"\nLoading model...")
        model = load_pretrained_vit(args.model_path, device)
        print(f"  Architecture : torchvision vit_l_16 (modified for {N_CHANNELS}-channel input)")
        print(f"  Input resize : {RESIZE_SIZE}×{RESIZE_SIZE}")

        # Evaluate
        start = time.time()
        evaluate(model, data_loader, device, args.subset, args.output_path)
        print(f"Total time: {(time.time() - start) / 60:.1f} minutes")

    except FileNotFoundError as e:
        print(f"✗ File error: {e}")
    except ValueError as e:
        print(f"✗ Validation error: {e}")
    except Exception as e:
        import traceback
        print(f"✗ Unexpected error: {type(e).__name__}: {e}")
        traceback.print_exc()
