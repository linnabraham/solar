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
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser(
        description="Plot a single-frame AIA image with IG attribution contour overlay."
    )
    parser.add_argument("--json-path",   default="solar_dataset.json")
    parser.add_argument("--model-path",  default="outputs/glad-shape-197/trained_model.pth",
                        help="Path to trained ViT model checkpoint.")
    parser.add_argument("--aarp-id",     type=int, default=3563,
                        help="AARP region identifier.")
    parser.add_argument("--t-idx",       type=int, default=12,
                        help="Time-step index within the AARP sequence.")
    parser.add_argument("--channel",     type=int, default=1,
                        help="Passband channel index (default: 1 = 131 Å).")
    parser.add_argument("--output-path", default=None,
                        help="Output PNG path. Defaults to "
                             "plots/contour_grid/<run-id>_aarp<aarp-id>.png.")
    args = parser.parse_args()

    if args.output_path is None:
        run_id = Path(args.model_path).parent.name
        args.output_path = f"plots/contour_grid/{run_id}_aarp{args.aarp_id}.png"

    config = TrainingConfig(json_path=args.json_path, stats_file="stats.pkl")
    config.trained_model_path = args.model_path
    metadata, model, transform, device = get_data_model(config)
    training_df, val_df, test_df = dfs_from_metadata(metadata)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    aarp_id = args.aarp_id
    s_images, attributions = run_pred_and_ig(aarp_id, val_df, transform, model, device)
    attributions_arr = np.array(attributions)
    channel = args.channel
    passband_attributions = attributions_arr[:, channel, :, :]
    t_idx = args.t_idx
    saliency = passband_attributions[t_idx]
    num_levels = 5
    contour_levels = np.linspace(np.min(saliency), np.max(saliency), num=num_levels+2)[1:-1]
    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    plot_aia_image(s_images[t_idx, channel, :, :], passband=131, vmax_percentile=99.9)
    plt.contour(saliency, levels=contour_levels[-1:], colors='red', linewidths=1.5)
    plt.savefig(args.output_path, bbox_inches="tight")
    print(f"Saved to {args.output_path}")
