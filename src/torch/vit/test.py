import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import argparse
import time
from dataclasses import dataclass
from typing import Tuple, Dict, List, Optional
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from aarp_ml.torch.dataset import aia_euv
from tqdm import tqdm
from torchvision.transforms import v2
from src.torch.vit.train import TrainingConfig
from src.torch.vit.utils import get_data_model
from ml_utils.visualization import plot_confusion_matrix

# ==================== Module Constants ====================
DEFAULT_BATCH_SIZE = 32
CONFUSION_MATRIX_CLASSES = [0, 1]
OUTPUT_PATH = "plots/cm_{subset}.png"
VALID_SUBSETS = {'training', 'validation', 'test'}
DEFAULT_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

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

    def __post_init__(self):
        """Validate configuration after initialization.

        Raises:
            ValueError: If subset is not valid.
        """
        if self.subset not in VALID_SUBSETS:
            raise ValueError(
                f"subset must be one of {VALID_SUBSETS}, got '{self.subset}'"
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
            - 'TN', 'FP', 'FN', 'TP': Confusion matrix elements
            - 'precision': TP / (TP + FP), or 0.0 if denominator is 0
            - 'recall': TP / (TP + FN), or 0.0 if denominator is 0
    
    Raises:
        ValueError: If confusion matrix is not 2x2 (binary classification only).
    """
    if cm.shape != (2, 2):
        raise ValueError(f"Expected 2x2 confusion matrix, got shape {cm.shape}")
    
    TN, FP, FN, TP = cm.ravel()
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    
    return {
        'TN': int(TN),
        'FP': int(FP),
        'FN': int(FN),
        'TP': int(TP),
        'precision': float(precision),
        'recall': float(recall),
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
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    
    n_total = len(val_dl.dataset)
    accuracy = (metrics['TP'] + metrics['TN']) / n_total
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"{'='*50}\n")

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
        choices=list(VALID_SUBSETS),
        help="Data subset to evaluate: training, validation, or test"
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
    
    args = parser.parse_args()

    if args.output_path is None:
        from pathlib import Path
        run_id = Path(args.trained_model).parent.name
        args.output_path = f"plots/cm_{run_id}_{args.subset}.png"

    try:
        # Load configuration
        print(f"Loading model from {args.trained_model}...")
        config = TrainingConfig(
            json_path=args.json_path,
            stats_file=args.stats_file
        )
        config.trained_model_path = args.trained_model
        
        # Load model and transform
        metadata, model, transform, device = get_data_model(config)
        device = torch.device(args.device if 'device' in args else DEFAULT_DEVICE)
        model.to(device)
        print(f"Using device: {device}")
        
        # Load dataset
        print(f"Loading {args.subset} dataset...")
        dataset = aia_euv(
            args.json_path,
            subset=args.subset,
            transform=v2.Compose([transform])
        )
        data_loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False
        )
        
        # Create confusion matrix config
        cm_config = ConfusionMatrixConfig(
            batch_size=args.batch_size,
            output_path=args.output_path,
            subset=args.subset,
            device=str(device)
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
