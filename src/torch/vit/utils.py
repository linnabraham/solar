import json
import torch
import gc
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from typing import List
from aarp_ml.dataset import all_wavelengths
from aarp_ml.torch.model import DeepFlare_ViT
from aarp_ml.torch.dataset import AIALogTransform
from src.torch.vit.train import TrainingConfig
from src.torch.vit.ig import do_ig


def get_metadata(config:TrainingConfig):
    with open(config.json_path, 'r') as json_file:
        metadata = json.load(json_file)
    return metadata

def get_metadata_from_json(json_path):
    with open(json_path, 'r') as json_file:
        metadata = json.load(json_file)
    return metadata

def get_attribution_for_image(images:np.array, label, transform, device, model, ib_size=1):
    """ Get Integrated Gradients attribution for a single image."""
    if images.ndim != 3:
        raise ValueError("Expecting a single timestep image and not a sequence")
    tensor_images = torch.from_numpy(images).to(torch.float32)
    tensor_data = transform(tensor_images)
    tensor_data = tensor_data.to(device)
    # apply the transform function on the zero baseline image as well to get
    # the proper zero baseline image
    baseline_zero = transform(torch.zeros_like(tensor_images))
    baseline_zero = baseline_zero.to(device)
    with torch.no_grad():
        ig_b0 = do_ig(tensor_data.unsqueeze(0), baseline=baseline_zero.unsqueeze(0), label=label, ib_size=ib_size, model=model)
    del tensor_images, tensor_data
    gc.collect()
    torch.cuda.empty_cache()
    return ig_b0

def get_model_and_transform(config):
    with open(config.stats_file, 'rb') as pickle_file:
        stats_data = pickle.load(pickle_file)
    means = [stats_data.get('mean').get(f'channel_{i}') for i in config.channel_indices]
    stds = [stats_data.get('std').get(f'channel_{i}') for i in config.channel_indices]

    transform = AIALogTransform(means=means, stds=stds)

    model = DeepFlare_ViT(height=config.image_height, n_classes=config.n_classes,
                          n_passbands=config.n_channels).model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(config.trained_model_path, map_location=device)
    learning_rate = config.learning_rate
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', -1) + 1
    else:
        model.load_state_dict(checkpoint)

    return model, transform, device

def get_data_model(config):
    metadata = get_metadata(config)
    model, transform, device = get_model_and_transform(config)
    return metadata, model, transform, device

def dfs_from_metadata(metadata):
    training_df = pd.DataFrame(metadata['training'])
    val_df = pd.DataFrame(metadata['validation'])
    test_df = pd.DataFrame(metadata['test'])
    for df in [training_df, val_df, test_df]:
        if not df.empty:
            df['timestamp'] = df['timestamp'].apply(pd.to_datetime).dt.tz_localize(None)
    training_df = training_df.rename({str(i):all_wavelengths[i] for i in range(7)}, axis=1)
    val_df = val_df.rename({str(i):all_wavelengths[i] for i in range(7)}, axis=1)
    test_df = test_df.rename({str(i):all_wavelengths[i] for i in range(7)}, axis=1)
    return training_df, val_df, test_df

def save_multi_channel_tensor_as_figure(
    image_tensor: torch.Tensor,
    filename: str,
    title: str,
    channel_labels: List[str],
    is_transformed: bool = False,
    means: List[float] = None,
    stds: List[float] = None,
    is_attribution_map: bool = False
):
    """
    Converts a multi-channel PyTorch tensor into a multi-panel figure 
    and saves it to a file.

    Args:
        image_tensor: The input tensor (expected shape [1, C, H, W] or [C, H, W]).
        filename: The name of the file to save the figure to (e.g., 'input_image.png').
        title: The overall title of the figure.
        channel_labels: List of labels for each channel (e.g., ['94 Å', '131 Å', ...]).
        is_transformed: If True, the tensor will be inverse-transformed 
                        using the provided means and stds (assumes natural log).
        means: List of mean values for inverse transformation. Required if is_transformed=True.
        stds: List of std deviation values for inverse transformation. Required if is_transformed=True.
        is_attribution_map: If True, uses a diverging colormap ('seismic') centered at 0.
    """
    # 1. Prepare the tensor
    # Ensure it's on CPU and remove the batch dimension (if present)
    data_np = image_tensor.detach().cpu().squeeze().numpy()

    if data_np.ndim != 3:
        raise ValueError(f"Input tensor must be 3D [C, H, W] after squeezing. Got shape: {data_np.shape}")

    C, H, W = data_np.shape

    # 2. Inverse Transform (if requested)
    if is_transformed:
        if means is None or stds is None:
            raise ValueError("Means and stds must be provided for inverse transformation.")

        # Reshape means and stds for broadcasting
        means_np = np.array(means)[:, None, None]
        stds_np = np.array(stds)[:, None, None]

        # Inverse Z-score: x_log = (x_norm * std) + mean
        x_log = (data_np * stds_np) + means_np

        # Inverse natural log: original_intensity = exp(x_log)
        data_np = np.exp(x_log)

        # Adjust title
        title = title + " (Inverse Transformed)"

    # 3. Plotting Setup
    rows = int(np.ceil(np.sqrt(C)))
    cols = int(np.ceil(C / rows))

    fig, axes = plt.subplots(rows, cols, figsize=(3.5 * cols, 3.5 * rows))
    axes = axes.flatten()

    # Determine Colormap and Normalization
    if is_attribution_map:
        # Use a diverging colormap for attributions (positive/negative scores)
        cmap = 'seismic'
        # Normalize around zero
        vmax = np.abs(data_np).max() * 0.99
        vmin = -vmax
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
    else:
        # Use a sequential colormap for intensity data
        cmap = 'magma'
        # Normalize based on percentiles for intensity data (to handle outliers)
        vmin = np.percentile(data_np, 1)
        vmax = np.percentile(data_np, 99.9)
        norm = plt.Normalize(vmin=vmin, vmax=vmax)

    # 4. Generate Subplots
    for i in range(C):
        ax = axes[i]
        im = ax.imshow(data_np[i], cmap=cmap, norm=norm)

        # Add a colorbar for each subplot (or one main one)
        # Using a single colorbar is cleaner for shared normalization

        ax.set_title(channel_labels[i], fontsize=10)
        ax.axis('off')

    # Hide unused subplots
    for i in range(C, len(axes)):
        fig.delaxes(axes[i])

    # Add a main colorbar
    if is_attribution_map:
        cbar_label = 'Shapley Value'
    elif is_transformed:
        cbar_label = 'Original Intensity (Data Number)'
    else:
        cbar_label = 'Normalized Value'

    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7]) # [left, bottom, width, height]
    fig.colorbar(im, cax=cbar_ax, label=cbar_label)

    # 5. Save Figure
    plt.tight_layout(rect=[0, 0, 0.9, 1]) # Adjust layout for colorbar
    fig.suptitle(title, y=1.02, fontsize=14)
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"Figure saved to '{filename}'.")

    return filename
