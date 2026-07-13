"""KernelSHAP attribution analysis for solar flare prediction ViT model.

Computes per-channel KernelSHAP attributions for training dataset samples and saves
aggregated statistics (mean importance + standard errors) to JSON for downstream analysis.

Typical usage:
    python kshap.py
        --trained_model_path outputs/glad-shape-197/trained_model.pth
        --json_path solar_dataset.json
        --stats_file stats.pkl
        --output_dir ./
        --num_samples 50
"""

from captum.attr import KernelShap
from aarp_ml.torch.model import DeepFlare_ViT
import vit_pytorch
import torch
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import time
import warnings
from tqdm import tqdm

# Data loading imports
from torch.utils.data import DataLoader, WeightedRandomSampler, SubsetRandomSampler
import pickle
from collections import Counter
import math
from aarp_ml.torch.dataset import aia_euv, AIALogTransform
from torchvision.transforms import v2

# Visualization and output
import matplotlib.pyplot as plt
import numpy as np
import json
from src.torch.vit.utils import save_multi_channel_tensor_as_figure

# ==================== MODULE-LEVEL CONSTANTS ====================

DEFAULT_BATCH_SIZE: int = 32
DEFAULT_N_PASSBANDS: int = 7
DEFAULT_N_CLASSES: int = 2
DEFAULT_HEIGHT: int = 512
DEFAULT_AIA_CHANNELS: List[int] = [94, 131, 171, 193, 211, 304, 335]
DEFAULT_JSON_PATH: str = "solar_dataset.json"
DEFAULT_STATS_FILE: str = "stats.pkl"
DEFAULT_OUTPUT_DIR: str = "."
DEFAULT_TRAINED_MODEL_PATH: str = "outputs/glad-shape-197/trained_model.pth"
# Output paths are derived from subset in KShapConfig.__post_init__ when left empty.
DEFAULT_OUTPUT_JSON: str = ""
DEFAULT_IMAGES_DIR: str = ""

# KernelSHAP configuration
DEFAULT_N_SAMPLES: int = 500
DEFAULT_MAX_SAMPLES: int = 50
DEFAULT_SEED: int = 42

# Data augmentation (unused — augmentation is disabled during attribution; see get_data_loader)
DEFAULT_FLIP_PROBABILITY: float = 0.5

DEFAULT_SAVE_IMAGES: bool = False

# Channel label formatting
AIA_CHANNEL_PREFIX: str = "AIA_"


@dataclass
class KShapConfig:
    """Configuration for KernelSHAP attribution computation.
    
    Attributes:
        trained_model_path: Path to trained ViT model checkpoint.
        batch_size: Number of samples per batch.
        json_path: Path to solar dataset JSON file (must exist).
        stats_file: Path to pickled normalization statistics (must exist).
        output_dir: Directory for saving output files (JSON).
        images_dir: Directory for saving image outputs (created if missing).
        n_passbands: Number of AIA channels/passbands.
        n_classes: Number of output classes (binary classification).
        height: Input image height in pixels.
        aia_channels: List of AIA wavelength identifiers.
        n_samples: Number of KernelSHAP perturbation samples.
        max_samples: Maximum number of training samples to process.
        output_json: Filename for JSON output.
    
    Raises:
        ValueError: If json_path or stats_file do not exist.
        ValueError: If n_samples < 100 or max_samples < 1.
    """
    trained_model_path: str = DEFAULT_TRAINED_MODEL_PATH
    batch_size: int = DEFAULT_BATCH_SIZE
    json_path: str = DEFAULT_JSON_PATH
    stats_file: str = DEFAULT_STATS_FILE
    output_dir: str = DEFAULT_OUTPUT_DIR
    images_dir: str = DEFAULT_IMAGES_DIR
    n_passbands: int = DEFAULT_N_PASSBANDS
    n_classes: int = DEFAULT_N_CLASSES
    height: int = DEFAULT_HEIGHT
    aia_channels: List[int] = field(default_factory=lambda: DEFAULT_AIA_CHANNELS)
    n_samples: int = DEFAULT_N_SAMPLES
    max_samples: int = DEFAULT_MAX_SAMPLES
    output_json: str = DEFAULT_OUTPUT_JSON
    seed: int = DEFAULT_SEED
    subset: str = 'training'
    save_images: bool = DEFAULT_SAVE_IMAGES
    min_free_mb: int = 0  # 0 = disabled; set >0 to abort cleanly if free GPU memory drops below this
    channel_indices: List[int] = None  # indices into DEFAULT_AIA_CHANNELS; None = all 7, in order

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        import os

        # Derive n_passbands/aia_channels/channel_indices consistently -- required for
        # models trained on a channel subset (e.g. the 94+131A CV models), where FITS
        # reads, stats lookups, and the model's input channel count must all agree.
        if self.channel_indices is None:
            self.channel_indices = list(range(self.n_passbands))
        else:
            self.n_passbands = len(self.channel_indices)
            self.aia_channels = [DEFAULT_AIA_CHANNELS[i] for i in self.channel_indices]

        # Derive output paths from subset when not explicitly set.
        if not self.output_json:
            self.output_json = f"shap_stats_{self.subset}.json"
        if not self.images_dir:
            self.images_dir = f"plots/kshap/{self.subset}/images"

        if not os.path.exists(self.json_path):
            raise ValueError(f"JSON dataset file not found: {self.json_path}")
        if not os.path.exists(self.stats_file):
            raise ValueError(f"Statistics file not found: {self.stats_file}")
        if self.n_samples < 100:
            raise ValueError(f"n_samples must be >= 100, got {self.n_samples}")
        if self.max_samples < 0:
            raise ValueError(f"max_samples must be >= 0 (0 = all available balanced), got {self.max_samples}")
        if self.max_samples > 0 and self.max_samples % 2 != 0:
            raise ValueError(f"max_samples must be even for stratified sampling, got {self.max_samples}")
        if len(self.aia_channels) != self.n_passbands:
            raise ValueError(
                f"aia_channels length ({len(self.aia_channels)}) "
                f"must match n_passbands ({self.n_passbands})"
            )

        os.makedirs(self.images_dir, exist_ok=True)


def get_weighted_sampler(dataset) -> WeightedRandomSampler:
    """Create a weighted sampler that handles class imbalance.
    
    Computes inverse-frequency weights for each class and assigns proportional
    weights to samples to ensure balanced representation during training.
    
    Args:
        dataset: torch.utils.data.Dataset with `.labels` attribute.
    
    Returns:
        WeightedRandomSampler with class-balanced weights.
    
    Raises:
        ValueError: If dataset has no labels attribute.
    """
    if not hasattr(dataset, 'labels'):
        raise ValueError(f"Dataset missing 'labels' attribute")
    
    class_counts = Counter(dataset.labels)
    total_samples = sum(class_counts.values())
    
    # Inverse-frequency weighting: less frequent classes get higher weights
    class_weights = {cls: total_samples / count for cls, count in class_counts.items()}
    sample_weights = [class_weights[label] for label in dataset.labels]
    
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(dataset),
        replacement=True
    )


def get_model(config: KShapConfig, device: torch.device) -> torch.nn.Module:
    """Load pretrained ViT model from checkpoint.
    
    Loads DeepFlare_ViT architecture and restores state from checkpoint dict.
    Moves model to specified device and sets eval mode.
    
    Args:
        config: KShapConfig instance with trained_model_path.
        device: torch.device (cuda or cpu).
    
    Returns:
        torch.nn.Module in eval mode on specified device.
    
    Raises:
        FileNotFoundError: If checkpoint file not found.
        RuntimeError: If model architecture mismatch with checkpoint weights.
        KeyError: If checkpoint missing expected 'model_state_dict' field.
    """
    try:
        model = DeepFlare_ViT(
            height=config.height,
            n_classes=config.n_classes,
            n_passbands=config.n_passbands
        ).model
        
        checkpoint = torch.load(config.trained_model_path, map_location=device)
        
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                raise KeyError("Checkpoint dict missing 'model_state_dict' field")
        else:
            # Legacy checkpoint format: direct state dict
            model.load_state_dict(checkpoint)
        
        model = model.to(device)
        model.eval()
        return model
    
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Model checkpoint not found: {config.trained_model_path}") from e


def get_stratified_indices(
    labels: List[int],
    max_samples: int,
    seed: int,
) -> List[int]:
    """Pick a class-balanced, deterministic, without-replacement subset of dataset indices.

    Splits ``max_samples`` evenly between class 0 and class 1 and draws each half
    without replacement using a seeded RNG. The resulting index list is shuffled
    so iteration order is not class-blocked.

    TODO: this is a per-IMAGE draw, not a per-AARP draw -- AARPs contribute wildly
    different image counts, so this is biased toward AARPs with more timesteps and
    can miss most available AARPs (e.g. 25 random negative-class images on the
    training subset touch only ~20 of 86 distinct negative AARPs). An AARP-stratified
    version (round-robin across distinct AARPs per class, one image per AARP per
    round before taking a second from any) would be more representative. Deferred.

    Args:
        labels: List of integer class labels for each dataset item (0 or 1).
        max_samples: Total number of indices to draw. Must be even.
        seed: RNG seed for reproducibility.

    Returns:
        List of dataset indices of length ``max_samples``.

    Raises:
        ValueError: If either class has fewer than ``max_samples // 2`` items.
    """
    labels_arr = np.asarray(labels)
    pos_pool = np.where(labels_arr == 1)[0]
    neg_pool = np.where(labels_arr == 0)[0]

    # max_samples == 0 means use all available samples (balanced on minority class).
    if max_samples == 0:
        per_class = min(len(pos_pool), len(neg_pool))
    else:
        per_class = max_samples // 2

    if len(pos_pool) < per_class or len(neg_pool) < per_class:
        raise ValueError(
            f"Stratified sampling requires {per_class} per class; "
            f"got {len(pos_pool)} positive, {len(neg_pool)} negative."
        )

    rng = np.random.default_rng(seed)
    pos_idx = rng.choice(pos_pool, size=per_class, replace=False)
    neg_idx = rng.choice(neg_pool, size=per_class, replace=False)
    indices = np.concatenate([pos_idx, neg_idx])
    rng.shuffle(indices)
    return indices.tolist()


def get_data_loader(
    config: KShapConfig,
    means: List[float],
    stds: List[float]
) -> DataLoader:
    """Create deterministic, class-balanced DataLoader for the training dataset.

    Applies only the log-normalization transform (no augmentation): explainability
    inputs must be deterministic. Uses :func:`get_stratified_indices` to draw an
    exact, without-replacement, class-balanced subset of size ``config.max_samples``.

    Args:
        config: KShapConfig instance with batch_size, json_path, max_samples, seed.
        means: Per-channel normalization means.
        stds: Per-channel normalization standard deviations.

    Returns:
        torch.utils.data.DataLoader iterating over exactly ``config.max_samples``
        unique training samples (half positive, half negative).

    Raises:
        FileNotFoundError: If json_path not found.
        ValueError: If means/stds length != n_passbands, or if a class has fewer
            than ``max_samples // 2`` items available.
    """
    if len(means) != config.n_passbands or len(stds) != config.n_passbands:
        raise ValueError(
            f"means/stds length ({len(means)}) must match "
            f"n_passbands ({config.n_passbands})"
        )

    train_dataset = aia_euv(
        config.json_path,
        subset=config.subset,
        transform=AIALogTransform(means, stds),
        channel_indices=config.channel_indices,
    )

    indices = get_stratified_indices(
        labels=train_dataset.labels,
        max_samples=config.max_samples,
        seed=config.seed,
    )

    # batch_size=1: KernelShap operates per-image, and the main loop processes
    # one sample per iteration. A larger batch would silently drop all but x[0].
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        sampler=SubsetRandomSampler(indices),
    )

    return train_loader


def save_images(
    image: torch.Tensor,
    baseline_mean: torch.Tensor,
    means: List[float],
    stds: List[float],
    filename_prefix: str,
    config: KShapConfig
) -> None:
    """Save normalized, original, and baseline images as multi-channel figures.

    Creates three PNG visualizations: normalized model input, original (denormalized),
    and mean baseline. Each image shows all AIA channels using save_multi_channel_tensor_as_figure.
    Images are saved to config.images_dir subdirectory.

    Args:
        image: Input tensor of shape (B, C, H, W) or (C, H, W).
        baseline_mean: Mean baseline tensor (same shape as image; all-zero in z-scored space).
        means: Per-channel normalization means.
        stds: Per-channel normalization standard deviations.
        filename_prefix: Prefix for output filenames (e.g., 'image_0').
        config: KShapConfig instance with aia_channels and images_dir.

    Raises:
        RuntimeError: If save_multi_channel_tensor_as_figure fails.
    """
    import os

    # Build full file paths in images_dir
    def get_filepath(suffix: str) -> str:
        return os.path.join(config.images_dir, f'{filename_prefix}_{suffix}.png')

    save_multi_channel_tensor_as_figure(
        image_tensor=image,
        filename=get_filepath('normalized'),
        title='Model Input Image (Normalized)',
        channel_labels=config.aia_channels,
    )

    save_multi_channel_tensor_as_figure(
        image_tensor=image,
        filename=get_filepath('original'),
        title='Input Image (Denormalized)',
        channel_labels=config.aia_channels,
        is_transformed=True,
        means=means,
        stds=stds
    )

    save_multi_channel_tensor_as_figure(
        image_tensor=baseline_mean,
        filename=get_filepath('baseline'),
        title='KernelSHAP Baseline (Mean Input)',
        channel_labels=config.aia_channels,
    )


def do_kernel_shap(
    config: KShapConfig,
    image: torch.Tensor,
    baseline_mean: torch.Tensor,
    true_label: int,
    model: torch.nn.Module
) -> Tuple[Dict[int, float], Dict[int, float]]:
    """Compute per-channel KernelSHAP attributions for an image.

    Uses Captum's KernelShap explainer with feature_mask to group pixels by channel.
    Returns aggregated mean attributions and standard errors per channel.

    Args:
        config: KShapConfig instance with n_samples, n_passbands, aia_channels.
        image: Input tensor of shape (1, C, H, W).
        baseline_mean: Mean baseline (all-zero in z-scored space) of same shape.
        true_label: True class label (0 or 1).
        model: torch.nn.Module in eval mode.

    Returns:
        Tuple of (channel_scores_mean, channel_scores_se) dicts mapping channel
        index (0-6) to float attribution values.

    Raises:
        RuntimeError: If KernelShap computation fails (GPU memory, etc.).
        ValueError: If image shape invalid (must be batch size 1).
    """
    if image.shape[0] != 1:
        raise ValueError(f"Expected batch size 1, got {image.shape[0]}")

    def wrapped_forward_fun(x: torch.Tensor) -> torch.Tensor:
        """Wrapper for model inference."""
        return model(x)

    # Create feature_mask grouping pixels by channel
    C = config.n_passbands
    feature_mask = torch.zeros_like(image, dtype=torch.long)
    for c in range(C):
        feature_mask[:, c, :, :] = c

    # Compute KernelSHAP attributions
    explainer = KernelShap(wrapped_forward_fun)
    attrs = explainer.attribute(
        image,
        baselines=baseline_mean,
        feature_mask=feature_mask,
        n_samples=config.n_samples,
        target=1,
        show_progress=True
    )
    
    # Aggregate attributions over spatial dimensions and batch
    channel_scores_mean = {}
    channel_scores_se = {}
    
    with torch.no_grad():
        per_c_mean = attrs.mean(dim=(0, 2, 3))  # Mean over batch, height, width
        per_c_std = attrs.std(dim=(0, 2, 3))    # Std over batch, height, width
        
        # Standard error = std / sqrt(n_observations)
        batch_size, _, height, width = attrs.shape
        n_obs = batch_size * height * width
        per_c_se = per_c_std / math.sqrt(n_obs)
        
        for c in range(C):
            channel_scores_mean[c] = float(per_c_mean[c].item())
            channel_scores_se[c] = float(per_c_se[c].item())
    
    return channel_scores_mean, channel_scores_se


def main() -> None:
    """Main execution: compute KernelSHAP stats on training samples and save JSON.
    
    Loads model, iterates over training batches with class balancing, computes
    per-channel KernelSHAP attributions for each sample, and saves aggregated
    statistics to shap_stats.json for downstream analysis. Images are saved to
    config.images_dir.
    
    Raises:
        FileNotFoundError: If required files (model, dataset, stats) not found.
        RuntimeError: If GPU memory or computation errors occur.
    """
    import argparse
    parser = argparse.ArgumentParser(description="KernelSHAP attribution for ViT solar-flare model")
    parser.add_argument("--trained-model-path", type=str, default=DEFAULT_TRAINED_MODEL_PATH,
                        help="Path to trained ViT checkpoint")
    parser.add_argument("--json-path", type=str, default=DEFAULT_JSON_PATH,
                        help="Path to dataset JSON (e.g. a CV fold's JSON for fold-specific checkpoints)")
    parser.add_argument("--max-samples", type=int, default=DEFAULT_MAX_SAMPLES,
                        help="Samples to process (must be even); 0 = all available, balanced on minority class")
    parser.add_argument("--subset", type=str, default="training",
                        choices=["training", "validation", "test"],
                        help="Dataset subset to draw samples from")
    parser.add_argument("--output-json", type=str, default="",
                        help="Output JSON path (default: shap_stats_{subset}.json)")
    parser.add_argument("--save-images", action="store_true",
                        help="Save per-sample visualization PNGs (disabled by default)")
    parser.add_argument("--min-free-mb", type=int, default=0,
                        help="Abort cleanly (saving partial results) if free GPU memory drops "
                             "below this many MB. 0 (default) disables the check -- set this "
                             "when running alongside another GPU job.")
    parser.add_argument("--channels", type=int, nargs="+", default=None,
                        help="AIA passbands the checkpoint was trained on, e.g. --channels 94 131. "
                             f"Choices: {DEFAULT_AIA_CHANNELS}. Default: all 7, in wavelength order.")
    parser.add_argument("--stats-file", type=str, default=DEFAULT_STATS_FILE,
                        help="Path to normalization stats pickle (e.g. stats_raw.pkl for pre-fix models)")
    args = parser.parse_args()

    channel_indices = None
    if args.channels is not None:
        unknown = [c for c in args.channels if c not in DEFAULT_AIA_CHANNELS]
        if unknown:
            parser.error(f"Unknown channel(s) {unknown}. Choices: {DEFAULT_AIA_CHANNELS}")
        channel_indices = [DEFAULT_AIA_CHANNELS.index(c) for c in args.channels]

    config = KShapConfig(
        trained_model_path=args.trained_model_path,
        json_path=args.json_path,
        max_samples=args.max_samples,
        subset=args.subset,
        output_json=args.output_json,
        channel_indices=channel_indices,
        save_images=args.save_images,
        min_free_mb=args.min_free_mb,
        stats_file=args.stats_file,
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Device: {device}")
    print(f"Images directory: {config.images_dir}")
    print(f"Config: {config}")
    
    
    # Load normalization statistics
    try:
        with open(config.stats_file, 'rb') as f:
            stats = pickle.load(f)
            means = [stats['mean'][f'channel_{i}'] for i in config.channel_indices]
            stds = [stats['std'][f'channel_{i}'] for i in config.channel_indices]
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Statistics file not found: {config.stats_file}") from e
    except KeyError as e:
        raise KeyError(f"Statistics dict missing expected channel keys: {e}") from e
    
    # Load model and data
    model = get_model(config, device)
    data_loader = get_data_loader(config, means, stds)
    transform = AIALogTransform(means, stds)

    # Process samples — loader yields exactly as many batches as indices selected.
    all_results = []
    n_total = len(data_loader)
    t_loop_start = time.perf_counter()

    for i, (x, y) in enumerate(tqdm(data_loader, desc="KernelSHAP", unit="sample")):
        if config.min_free_mb > 0 and device.type == 'cuda':
            free_mb = torch.cuda.mem_get_info()[0] / (1024 ** 2)
            if free_mb < config.min_free_mb:
                print(f"\n! Free GPU memory {free_mb:.0f}MB < --min-free-mb {config.min_free_mb}MB "
                      f"at sample {i}/{n_total}. Stopping early and saving partial results.")
                break

        image = x.to(device)  # already shape (1, C, H, W) since batch_size=1
        # Mean baseline: image is already log+z-scored, so an all-zero tensor here
        # IS "every channel at its typical (mean) value" -- NOT transform(zeros),
        # which would clamp raw-zero to 1 DN and z-score that (the dark/near-black
        # baseline). The dark baseline's distance-from-baseline for a typical pixel
        # is mean/std per channel, which ranges ~3x (94A) to ~18x (193A) just from
        # each channel's own scale -- a baseline-choice artifact, not real signal.
        baseline_mean = torch.zeros_like(image).to(device)

        if config.save_images:
            save_images(image, baseline_mean, means, stds, f"image_{i}", config)

        true_label = int(y.item())

        # Forward pass to record the model's prediction for this sample.
        # Saved alongside the SHAP record so analyze_shap.py can filter to
        # correctly-classified samples without recomputing.
        with torch.no_grad():
            prediction = int(model(image).argmax(dim=1).item())

        # Compute KernelSHAP attributions w.r.t. the true class
        mean_scores, se_scores = do_kernel_shap(config, image, baseline_mean, true_label, model)

        record = {
            "round": i,
            "label": true_label,
            "prediction": prediction,
            "importance": {
                f"{AIA_CHANNEL_PREFIX}{config.aia_channels[c]}": score
                for c, score in mean_scores.items()
            },
            "std_error": {
                f"{AIA_CHANNEL_PREFIX}{config.aia_channels[c]}": se
                for c, se in se_scores.items()
            }
        }
        all_results.append(record)

        elapsed = time.perf_counter() - t_loop_start
        per_sample = elapsed / (i + 1)
        print(f"  sample {i+1}/{n_total} | {per_sample:.1f}s/sample | "
              f"label={true_label} pred={prediction}")
    
    # Save all results to JSON
    output_path = f"{config.output_dir}/{config.output_json}"
    try:
        with open(output_path, 'w') as f:
            json.dump(all_results, f, indent=4)
        print(f"✓ KernelSHAP statistics saved to {output_path}")
    except IOError as e:
        raise IOError(f"Failed to write output JSON: {output_path}") from e


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, RuntimeError, KeyError, IOError) as e:
        print(f"✗ Error: {e}")
        raise
