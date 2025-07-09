import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import math
import torch
import gc
from itertools import islice
from torch.utils.data import DataLoader, TensorDataset
from src.torch.vit.train import TrainingConfig
from src.torch.vit.utils import get_data_model, dfs_from_metadata
from src.torch.vit.ig import single_aarp, do_ig
from astro_utils.aia import plot_aia_image

def run_pred_and_ig(aarp_id, metadata_df, transform, model, device):
    aarp_id_df = metadata_df.query(f'aarp_id == {aarp_id}')
    s_aarp = single_aarp(aarp_id, aarp_id_df)
    s_images = s_aarp.get_images()
    # Use the model to make predictions
    tensor_images = torch.from_numpy(s_images).to(torch.float32)  # shape: [x, 7, 512, 512]
    tensor_data = transform(tensor_images)
    tensor_data = tensor_data.to(device)
    dataset = TensorDataset(tensor_data)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    attributions = []
    label = s_aarp.label
    ib_size = 1
    n_images = s_images.shape[0]
    with torch.no_grad():
        for (batch,) in islice(loader, n_images):
            batch = batch.to(device)
            baseline_zero = transform(torch.zeros_like(batch))
            baseline_zero = baseline_zero.to(device)
            ig_b0 = do_ig(batch, baseline_zero, label=label, ib_size=ib_size, model=model)
            attributions.append(ig_b0)
            del batch, ig_b0
            torch.cuda.empty_cache()
            gc.collect()
    del tensor_images, tensor_data, dataset
    return s_images, attributions

if __name__=="__main__":
    config = TrainingConfig(json_path="solar_dataset.json", stats_file="stats.pkl")
    config.trained_model_path = "output/glad-shape-197/trained_model.pth"
    metadata, model, transform, device = get_data_model(config)
    training_df, val_df, test_df = dfs_from_metadata(metadata)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    aarp_id = 3563
    s_images, attributions = run_pred_and_ig(aarp_id, val_df, transform, model, device)
    attributions_arr = np.array(attributions)
    channel = 1
    passband_attributions = attributions_arr[:, channel, :,:]
    t_idx = 12
    saliency = passband_attributions[t_idx]
    num_levels = 5
    contour_levels = np.linspace(np.min(saliency), np.max(saliency), num=num_levels+2)[1:-1]
    plot_aia_image(s_images[t_idx,channel,:,:], passband=131, vmax_percentile=99.9)
    plt.contour(saliency, levels=contour_levels[-1:], colors='red', linewidths=1.5)
    plt.savefig("tests_outputs/contour_grid.png", bbox_inches="tight")
