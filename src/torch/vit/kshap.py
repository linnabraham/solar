from captum.attr import KernelShap
from aarp_ml.torch.model import DeepFlare_ViT
import vit_pytorch
import torch
from dataclasses import dataclass
# imports for dataloader
from torch.utils.data import DataLoader
import pickle
from torch.utils.data import WeightedRandomSampler
from collections import Counter
import math
from aarp_ml.torch.dataset import aia_euv, AIALogTransform
from torchvision.transforms import v2
import matplotlib.pyplot as plt
import numpy as np
import json
from src.torch.vit.utils import save_multi_channel_tensor_as_figure

def get_weighted_sampler(dataset) -> WeightedRandomSampler:
    """Create a sampler that handles class imbalance."""
    # Count instances of each class
    class_counts = Counter(dataset.labels)
    total_samples = sum(class_counts.values())
    # Compute class weights (inverse of frequency)
    class_weights = {cls: total_samples / count for cls, count in class_counts.items()}
    # Assign a weight to each sample based on its class
    sample_weights = [class_weights[label] for label in dataset.labels]
    return WeightedRandomSampler(weights=sample_weights, num_samples=len(dataset), replacement=True)

@dataclass
class Config:
    trained_model_path: str = None
    batch_size = 32
    json_path = "solar_dataset.json"
    stats_file = "stats.pkl"
    n_passbands = 7
    n_classes = 2
    height = 512
    aia_channels = [94, 131, 171, 193, 211, 304, 335]
    output_dir = "."

def get_model(config:Config, device:torch.device):

    model = DeepFlare_ViT(
      height=config.height,
      n_classes=config.n_classes,
      n_passbands=config.n_passbands
    ).model

    checkpoint = torch.load(config.trained_model_path, map_location=device)
    learning_rate = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', -1) + 1
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()
    return model

def get_data_loader(config:Config, means, stds):

    train_dataset = aia_euv(
        config.json_path,
        subset='training',
        transform=v2.Compose([
            AIALogTransform(means, stds),
            #v2.Resize((224, 224)),          # <--- add this line
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5)
        ])
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        sampler=get_weighted_sampler(train_dataset)
    )

    return train_loader

def save_images(image, baseline_zero, means, stds, filename_prefix, config):

    save_multi_channel_tensor_as_figure(
        image_tensor=image,
        filename=f'{filename_prefix}_normalized.png',
        title='Model Input Image (Normalized)',
        channel_labels=config.aia_channels,
    )

    save_multi_channel_tensor_as_figure(
        image_tensor=image,
        filename=f'{filename_prefix}_original.png',
        title='Input Image',
        channel_labels=config.aia_channels,
        is_transformed=True, # Set this to True
        means=means,
        stds=stds
    )

    save_multi_channel_tensor_as_figure(
        image_tensor=baseline_zero,
        filename=f'{filename_prefix}_baseline.png',
        title='KernelSHAP Baseline (Zero Input)',
        channel_labels=config.aia_channels,
    )

def do_kernel_shap(config, image, baseline_zero, true_label, model):

    def wrapped_forward_fun(image):
        return model(image)

    print(wrapped_forward_fun(image).shape)

    C = config.n_passbands
    # Group all pixels of each channel as ONE feature via a feature_mask
        # feature_mask must be same shape as x, with integer IDs for groups.
        # We'll label each channel with a distinct ID 0..C-1.
    feature_mask = torch.zeros_like(image, dtype=torch.long)
    for c in range(C):
        feature_mask[:, c, :, :] = c
    if image.shape[0] != 1:
        feature_mask = feature_mask[0][None,...]

    explainer = KernelShap(wrapped_forward_fun)
    n_samples = 500
    attrs = explainer.attribute(
            image,
            baselines=baseline_zero,
            feature_mask=feature_mask,
            n_samples=n_samples,
            target=true_label,
            show_progress=True
            )

    # Aggregate to per-channel Shapley:
    # Captum returns per-element attributions; sum or mean over H,W (and batch).
    # Shapley is additive; sum is a natural aggregation. Mean gives scale-less scores.
    # We'll use mean over spatial & batch for comparability across image sizes.
    channel_scores_mean = {}
    channel_scores_se = {}
    with torch.no_grad():
        # reduce over B,H,W
        # You can also use .abs() if you want magnitude-only importance.
        per_c = attrs.mean(dim=(0, 2, 3))  # (C,)
        shp = attrs.shape
        per_c_se = attrs.std(dim=(0, 2, 3)) /  math.sqrt(shp[0]*shp[2]*shp[3])# (C,)
        for c in range(C):
            channel_scores_mean[c] = float(per_c[c].item())
            channel_scores_se[c] = float(per_c_se[c].item())

    return channel_scores_mean, channel_scores_se

def main():
    config = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config.trained_model_path = "outputs/glad-shape-197/trained_model.pth"

    # Load statistics
    with open(config.stats_file, 'rb') as f:
        stats = pickle.load(f)
        means = [stats['mean'][f'channel_{i}'] for i in range(config.n_passbands)]
        stds = [stats['std'][f'channel_{i}'] for i in range(config.n_passbands)]

    train_loader = get_data_loader(config, means, stds)

    model = get_model(config, device)

    #train_iter = iter(train_loader)

    #x, y = next(train_iter)

    transform = AIALogTransform(means, stds)

    all_results = []

    for i, (x, y) in enumerate(train_loader):

        if i >= 50:
            break

        image = x[0].unsqueeze(0)
        image = image.to(device)
        baseline_zero = transform(torch.zeros_like(image))
        save_images(image, baseline_zero, means, stds, f"image_{i}", config)

        true_label = y[0].unsqueeze(0).item()
        mean_scores, se_scores = do_kernel_shap(config, image, baseline_zero, true_label, model)
        print(channel_scores_mean, channel_scores_se)

        # Create a record for this round
        # Map indices (0-6) to actual AIA channel names (94, 131...)
        record = {
            "round": i,
            "label": true_label,
            "importance": {
                f"AIA_{config.aia_channels[c]}": score 
                for c, score in mean_scores.items()
            },
            "std_error": {
                f"AIA_{config.aia_channels[c]}": se 
                for c, se in se_scores.items()
            }
        }

        all_results.append(record)

    # Save everything at the end
    #json_path = os.path.join(config.output_dir, "shap_stats.json")
    json_path = "shap_stats.json"
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=4)

    print(f"Stats saved to {json_path}")
if __name__=="__main__":
    main()
