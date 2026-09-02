import torch
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
import argparse
import time
from dataclasses import dataclass
from typing import Tuple, Dict, List, Optional
from torch.utils.data import DataLoader, ConcatDataset
from sklearn.metrics import confusion_matrix
from aarp_ml.torch.dataset import aia_euv
from aarp_ml.dataset import all_wavelengths
from tqdm import tqdm
from torchvision.transforms import v2
from src.torch.vit.train import TrainingConfig
from src.torch.vit.utils import get_data_model, needs_resize
from ml_utils.visualization import plot_confusion_matrix

# ==================== Module Constants ====================
DEFAULT_BATCH_SIZE = 32
CONFUSION_MATRIX_CLASSES = [0, 1]
OUTPUT_PATH = "plots/cm_{subset}.png"
VALID_SUBSETS = {'training', 'validation', 'test'}
DEFAULT_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
VALID_MODEL_TYPES = {"vit": "deepflare_vit", "vit-pretrained": "vit_pretrained"}

# ==================== Configuration Class ====================
@dataclass
class ConfusionMatrixConfig:
    """Configuration for confusion matrix computation and visualization.
    
    Attributes:
        batch_size (int): Batch size for data loading. Defaults to 32.
        output_path (str): Path where confusion matrix plot will be saved. Defaults to "plots/cm.png".
        subset (str): Data subset to evaluate ('training', 'validation', or 'test'). Defaults to 'validation'.
        device (str): Device to use for model ('cuda' or 'cpu'). Defaults to auto-detected.
    """
    batch_size: int = DEFAULT_BATCH_SIZE
    output_path: str = None
    subset: str = 'validation'
    device: str = DEFAULT_DEVICE
    metrics_path: Optional[str] = None  # if set, write metrics dict as JSON here

    def __post_init__(self):
        """Validate configuration after initialization.

        Accepts single subset names ('test', 'validation', 'training') or
        combined labels like 'validation+test' produced when multiple subsets
        are concatenated.

        Raises:
            ValueError: If any component of the subset label is not valid.
        """
        components = self.subset.split("+")
        invalid = [c for c in components if c not in VALID_SUBSETS]
        if invalid:
            raise ValueError(
                f"subset components must be one of {VALID_SUBSETS}, got {invalid}"
            )
        if self.output_path is None:
            self.output_path = OUTPUT_PATH.format(subset=self.subset)

# ==================== Metrics Helper Function ====================
def compute_metrics(cm: np.ndarray) -> Dict[str, float]:
    """Extract classification metrics from confusion matrix.
    
    Computes True Positives, False Positives, False Negatives, True Negatives,
    as well as precision, recall, and overall accuracy. Handles division by zero
    gracefully by returning 0.0 for undefined metrics.
    
    Args:
        cm (np.ndarray): 2D confusion matrix array of shape (2, 2) for binary classification.
            Should be in the format: [[TN, FP], [FN, TP]].
    
    Returns:
        Dict[str, float]: Dictionary containing:
            - 'TN', 'FP', 'FN', 'TP', 'n_samples': confusion matrix elements and total
            - 'base_rate': positive-class fraction (TP + FN) / n_samples
            - 'recall', 'specificity', 'balanced_accuracy': base-rate-insensitive metrics
            - 'precision', 'accuracy': base-rate-sensitive metrics
            - 'tss', 'hss': skill scores (TSS = recall + specificity - 1)
            All ratio metrics return 0.0 when their denominator is 0.

    Raises:
        ValueError: If confusion matrix is not 2x2 (binary classification only).
    """
    if cm.shape != (2, 2):
        raise ValueError(f"Expected 2x2 confusion matrix, got shape {cm.shape}")

    TN, FP, FN, TP = (int(v) for v in cm.ravel())
    n_total = TN + FP + FN + TP
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    accuracy = (TP + TN) / n_total if n_total > 0 else 0.0
    balanced_accuracy = (recall + specificity) / 2
    tss = recall + specificity - 1.0
    hss_denominator = (TP + FP) * (FP + TN) + (TP + FN) * (FN + TN)
    hss = 2.0 * (TP * TN - FP * FN) / hss_denominator if hss_denominator > 0 else 0.0
    base_rate = (TP + FN) / n_total if n_total > 0 else 0.0

    return {
        'TN': TN,
        'FP': FP,
        'FN': FN,
        'TP': TP,
        'n_samples': n_total,
        'base_rate': float(base_rate),
        'recall': float(recall),
        'specificity': float(specificity),
        'balanced_accuracy': float(balanced_accuracy),
        'precision': float(precision),
        'accuracy': float(accuracy),
        'tss': float(tss),
        'hss': float(hss),
    }

# ==================== Main Evaluation Function ====================
def main(
    *,
    model: torch.nn.Module,
    val_dl: DataLoader,
    loss_func: torch.nn.Module,
    device: torch.device,
    config: ConfusionMatrixConfig
) -> None:
    """Evaluate model on validation/test set and generate confusion matrix visualization.
    
    Runs the model in evaluation mode over the data loader, collects predictions
    and true labels, computes confusion matrix and metrics, and saves a visualization.
    
    Args:
        model (torch.nn.Module): Trained neural network model in evaluation mode.
        val_dl (DataLoader): DataLoader for validation/test data (batches of images & labels).
        loss_func (torch.nn.Module): Loss function (e.g., CrossEntropyLoss) for computing validation loss.
        device (torch.device): Device to run model on ('cuda' or 'cpu').
        config (ConfusionMatrixConfig): Configuration including output path, batch size, etc.
    
    Returns:
        None. Saves confusion matrix visualization to config.output_path.
    
    Side effects:
        - Saves matplotlib figure to disk at config.output_path
        - Prints confusion matrix, metrics, and timing info to stdout
    
    Example:
        >>> config = ConfusionMatrixConfig(subset='validation', output_path='plots/cm.png')
        >>> main(model=model, val_dl=val_loader, loss_func=criterion, device=device, config=config)
    """
    model.eval()
    val_loss = 0.0
    y_true: List[int] = []
    y_pred: List[int] = []

    # Run inference on all batches
    with torch.inference_mode():
        for images, labels in tqdm(val_dl, leave=False, desc=f"Evaluating {config.subset} set"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            val_loss += loss_func(outputs, labels) * labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=CONFUSION_MATRIX_CLASSES)
    metrics = compute_metrics(cm)
    
    # Print results
    print(f"\n{'='*50}")
    print(f"Confusion Matrix ({config.subset} set):")
    print(f"{'='*50}")
    print(cm)
    
    print(f"\nMetrics:")
    print(f"  True Positives (TP): {metrics['TP']}")
    print(f"  False Positives (FP): {metrics['FP']}")
    print(f"  False Negatives (FN): {metrics['FN']}")
    print(f"  True Negatives (TN): {metrics['TN']}")
    print(f"  Recall (sensitivity): {metrics['recall']:.4f}")
    print(f"  Specificity: {metrics['specificity']:.4f}")
    print(f"  Balanced accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  TSS: {metrics['tss']:.4f}  HSS: {metrics['hss']:.4f}")
    print(f"  Base rate: {metrics['base_rate']:.4f} ({metrics['n_samples']} samples)")
    print(f"{'='*50}\n")

    if config.metrics_path is not None:
        metrics_record = {'subset': config.subset, **metrics}
        Path(config.metrics_path).parent.mkdir(parents=True, exist_ok=True)
        with open(config.metrics_path, 'w') as f:
            json.dump(metrics_record, f, indent=2)
        print(f"✓ Saved metrics to {config.metrics_path}")

    # Generate and save confusion matrix visualization
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
    
    Path(config.output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(config.output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"✓ Saved confusion matrix to {config.output_path}")

# ==================== Main Block ====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate trained model and generate confusion matrix."
    )
    parser.add_argument(
        "--subset",
        required=True,
        nargs="+",
        choices=list(VALID_SUBSETS),
        help="One or more subsets to evaluate: training, validation, test. "
             "Pass multiple to combine them (e.g. --subset validation test)."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Batch size for evaluation (default: {DEFAULT_BATCH_SIZE})"
    )
    parser.add_argument(
        "--output-path",
        default=None,
        help="Output path for confusion matrix plot (default: plots/cm_<subset>.png)"
    )
    parser.add_argument(
        "--trained-model",
        "--model-path",
        default="outputs/glad-shape-197/trained_model.pth",
        help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--json-path",
        default="solar_dataset.json",
        help="Path to dataset JSON file"
    )
    parser.add_argument(
        "--stats-file",
        default="stats.pkl",
        help="Path to stats pickle file"
    )
    parser.add_argument(
        "--channels",
        type=int,
        nargs="+",
        default=None,
        help="AIA passbands the model was trained on, e.g. --channels 94 131. "
             f"Choices: {all_wavelengths}. Default: all 7, in wavelength order."
    )
    parser.add_argument(
        "--metrics-out",
        default=None,
        help="If set, write the computed metrics as JSON to this path "
             "(recall, specificity, balanced accuracy, precision, accuracy, TSS, HSS, base rate)"
    )
    parser.add_argument(
        "--model-type",
        default="vit",
        choices=list(VALID_MODEL_TYPES),
        help="Model architecture: 'vit' (DeepFlare_ViT, default) or "
             "'vit-pretrained' (torchvision vit_l_16)."
    )

    args = parser.parse_args()

    channel_indices = None
    if args.channels is not None:
        unknown = [c for c in args.channels if c not in all_wavelengths]
        if unknown:
            parser.error(f"Unknown channel(s) {unknown}. Choices: {all_wavelengths}")
        channel_indices = [all_wavelengths.index(c) for c in args.channels]

    # Build a single label from the subset list, e.g. "validation+test"
    subset_label = "+".join(args.subset)

    if args.output_path is None:
        # TODO: for a --save-all-epochs checkpoint (outputs/<run>/epoch_checkpoints/epoch_NN.pth),
        # parent.name is always "epoch_checkpoints" -- every epoch tested this way collides on
        # the same plot filename. Should include the checkpoint filename stem (epoch_NN) too.
        run_id = Path(args.trained_model).parent.name
        args.output_path = f"plots/cm_{run_id}_{subset_label}.png"

    try:
        # Load configuration
        print(f"Loading model from {args.trained_model}...")
        config = TrainingConfig(
            json_path=args.json_path,
            stats_file=args.stats_file,
            channel_indices=channel_indices,
            model_type=VALID_MODEL_TYPES[args.model_type],
        )
        config.trained_model_path = args.trained_model

        # Load model and transform
        metadata, model, transform, device = get_data_model(config)
        print(f"Using device: {device}")
        resize_to = needs_resize(config)
        print(f"Model type: {args.model_type}" + (f"  (resize to {resize_to})" if resize_to else ""))

        # Load dataset — combine with ConcatDataset when multiple subsets given
        print(f"Loading {subset_label} dataset...")
        transform_steps = [transform] if resize_to is None else [transform, v2.Resize(resize_to)]
        datasets = [
            aia_euv(args.json_path, subset=s, transform=v2.Compose(transform_steps),
                    channel_indices=config.channel_indices)
            for s in args.subset
        ]
        dataset = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)
        data_loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False
        )
        print(f"  {len(dataset)} samples total")

        # Create confusion matrix config
        cm_config = ConfusionMatrixConfig(
            batch_size=args.batch_size,
            output_path=args.output_path,
            subset=subset_label,
            device=str(device),
            metrics_path=args.metrics_out
        )
        
        # Run evaluation
        criterion = torch.nn.CrossEntropyLoss()
        start_time = time.time()
        main(
            model=model,
            val_dl=data_loader,
            loss_func=criterion,
            device=device,
            config=cm_config
        )
        elapsed_minutes = (time.time() - start_time) / 60
        print(f"Total time: {elapsed_minutes:.1f} minutes")
        
    except FileNotFoundError as e:
        print(f"✗ File error: {e}")
    except ValueError as e:
        print(f"✗ Validation error: {e}")
    except Exception as e:
        print(f"✗ Unexpected error: {type(e).__name__}: {e}")
