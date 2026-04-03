"""Attribution analysis for solar flare prediction ViT model.

This module generates visualization and analysis plots for model attributions
(importance maps showing which image regions influenced predictions) on test and
validation data. Uses integrated gradients to compute per-sample attributions and
creates plots showing attribution percentiles and filtered image intensities.

Outputs:
    - attributions_percentiles_99.png: Line plot of 99th percentile attribution values
    - filtered_image_intensities.png: Boxplot of image intensities in high-attribution regions
"""

import os
import gc
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset
from src.torch.vit.utils import get_data_model, dfs_from_metadata
from src.torch.vit.predictions_analyze import get_aarp_seq_dataset
from src.torch.vit.train import TrainingConfig
from src.torch.vit.ig import single_aarp, run_ig
from src.torch.vit.class_wise_distribution import run_pred_and_ig

# Analysis parameters
CHANNEL_IDX: int = 1
"""Image channel index for single-passband analysis."""

PERCENTILE_LEVEL: int = 99
"""Percentile threshold for attribution visualization."""

NUM_CONTOUR_LEVELS: int = 5
"""Number of contour levels for saliency thresholding."""

BOXPLOT_SAMPLING_STEP: int = 10
"""Sample every Nth element for boxplot to avoid overcrowding."""

# Visualization parameters
OUTPUT_FIGSIZE: tuple = (12, 6)
"""Figure size for boxplot visualization."""

OUTPUT_DPI: int = 150
"""DPI resolution for saved output images."""

# Model and data parameters
DEFAULT_LEARNING_RATE: float = 0.001
"""Adam optimizer learning rate."""

TRAINED_MODEL_PATH: str = "outputs/glad-shape-197/trained_model.pth"
"""Path to trained ViT model checkpoint."""

OUTPUT_HOME: str = "plots/predictions"
"""Root directory for attribution output plots."""

def create_plots(
    aarp_id: int,
    metadata_df,
    transform,
    model: torch.nn.Module,
    device: torch.device,
    output_home: str = OUTPUT_HOME,
) -> None:
    """Generate and save attribution analysis plots for a single AARP event.

    Creates two visualization plots:
    1. Line plot showing 99th percentile attribution value across time steps
    2. Boxplot showing distribution of image intensities in high-attribution regions

    Args:
        aarp_id: Unique AARP event identifier
        metadata_df: DataFrame containing event metadata and labels
        transform: PyTorch transform pipeline for image preprocessing
        model: Trained ViT model for making predictions and computing attributions
        device: Torch device (cuda or cpu) for model inference
        output_home: Root directory for saving output plots (default: "plots/predictions")

    Returns:
        None. Saves PNG files to {output_home}/{aarp_id}/
    """
    output_dir = f"{output_home}/{aarp_id}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        print(f"Using existing output directory {output_dir}")

    print(f"Using {aarp_id=}")

    # Create and plot attributions for a single passband and single percentile level
    s_images, attributions = run_pred_and_ig(aarp_id, metadata_df, transform=transform, model=model, device=device)

    passband_attributions = [attribution[CHANNEL_IDX, :, :] for attribution in attributions]
    torch.cuda.empty_cache()
    gc.collect()

    flattened_attributions = [passband_attribution.flatten() for passband_attribution in passband_attributions]
    plt.plot([np.percentile(flattened_attribution, PERCENTILE_LEVEL) for flattened_attribution in flattened_attributions])
    plt.savefig(f"{output_dir}/attributions_percentiles_{PERCENTILE_LEVEL}.png", bbox_inches="tight", dpi=OUTPUT_DPI)
    plt.close()

    # Plot boxplot of intensities taken using the thresholded attribution mask for a single threshold and single passband
    passband_images = s_images[:, CHANNEL_IDX, :, :]
    filtered_image_intensities_list = []
    for t_idx in range(len(passband_attributions)):
        saliency = passband_attributions[t_idx]
        contour_levels = np.linspace(np.min(saliency), np.max(saliency), num=NUM_CONTOUR_LEVELS + 2)[1:-1]
        filtered_images = np.where(saliency > contour_levels[-1], passband_images[t_idx], 0)
        filtered_image_intensities_list.append(filtered_images[filtered_images > 0])
    plt.figure(figsize=OUTPUT_FIGSIZE)
    plt.boxplot(filtered_image_intensities_list[::BOXPLOT_SAMPLING_STEP])
    plt.savefig(f"{output_dir}/filtered_image_intensities.png", bbox_inches="tight", dpi=OUTPUT_DPI)
    plt.close()

    del attributions
    torch.cuda.empty_cache()
    gc.collect()

def main() -> None:
    """Load model, compute attributions, and generate analysis plots.

    Loads a trained ViT model and dataset, then iterates through all test and
    validation samples to compute integrated gradient attributions and save
    visualization plots showing which image regions most influenced the model's
    predictions.

    Raises:
        FileNotFoundError: If the trained model checkpoint does not exist
        RuntimeError: If model loading or initialization fails
    """
    # Load Data and Model
    config = TrainingConfig(json_path="solar_dataset.json", stats_file="stats.pkl")
    config.trained_model_path = TRAINED_MODEL_PATH
    metadata, model, transform, device = get_data_model(config)
    training_df, val_df, test_df = dfs_from_metadata(metadata)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(TRAINED_MODEL_PATH, map_location=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=DEFAULT_LEARNING_RATE)
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', -1) + 1
    else:
        model.load_state_dict(checkpoint)
    model = model.to(device)

    os.makedirs(OUTPUT_HOME, exist_ok=True)

    for aarp_id in test_df.aarp_id.unique().tolist():
        create_plots(aarp_id, test_df, transform, model, device, OUTPUT_HOME)
    for aarp_id in val_df.aarp_id.unique().tolist():
        create_plots(aarp_id, val_df, transform, model, device, OUTPUT_HOME)

if __name__=="__main__":
    main()
