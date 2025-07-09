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
from src.torch.vit.class_wise_distribution import run_pred_and_ig
from astro_utils.aia import plot_aia_image

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
