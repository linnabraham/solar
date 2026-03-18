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

def create_plots(aarp_id, metadata_df, transform, model, device, output_home):
    output_dir = f"{output_home}/{aarp_id}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        # print(f"Output directory {output_dir} already exists, skipping {aarp_id}")
        print(f"Using existing output directory {output_dir}")
        # return None

    print(f"Using {aarp_id=}")

    # Create and plot attributions for a single passband and single percentile level
    s_images, attributions = run_pred_and_ig(aarp_id, metadata_df, transform=transform, model=model, device=device)

    channel = 1
    passband_attributions = [attribution[channel, :, :] for attribution in attributions]
    torch.cuda.empty_cache()
    gc.collect()

    flattened_attributions = [passband_attribution.flatten() for passband_attribution in passband_attributions]
    percentile_level = 99
    plt.plot([ np.percentile(flattened_attribution, percentile_level) for flattened_attribution in flattened_attributions])
    plt.savefig(f"{output_dir}/attributions_percentiles_{percentile_level}.png", bbox_inches="tight", dpi=150)
    plt.close()

    # Plot boxplot of intensities taken using the thresholded attribution mask for a single threshold and single passband
    passband_images = s_images[:,channel,::]
    filtered_image_intensities_list = []
    for t_idx in range(len(passband_attributions)):
        saliency = passband_attributions[t_idx]
        num_levels = 5
        contour_levels = np.linspace(np.min(saliency), np.max(saliency), num=num_levels+2)[1:-1]
        filtered_images = np.where(saliency > contour_levels[-1], passband_images[t_idx], 0)
        filtered_image_intensities_list.append(filtered_images[filtered_images>0])
    plt.figure(figsize=(12,6))
    plt.boxplot(filtered_image_intensities_list[::10])
    plt.savefig(f"{output_dir}/filtered_image_intensities.png", bbox_inches="tight", dpi=150)
    plt.close()

    del attributions
    torch.cuda.empty_cache()
    gc.collect()

def main():
    # Load Data and Model
    config = TrainingConfig(json_path="solar_dataset.json", stats_file="stats.pkl")
    config.trained_model_path = "output/glad-shape-197/trained_model.pth"
    metadata, model, transform, device = get_data_model(config)
    training_df, val_df, test_df = dfs_from_metadata(metadata)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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

    output_home = "pred-output"
    os.makedirs(output_home, exist_ok=True)

    for aarp_id in test_df.aarp_id.unique().tolist():
        create_plots(aarp_id, test_df, transform, model, device, output_home)
    for aarp_id in val_df.aarp_id.unique().tolist():
        create_plots(aarp_id, val_df, transform, model, device, output_home)

if __name__=="__main__":
    main()
