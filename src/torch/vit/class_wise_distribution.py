import os
import gc
import json
from aarp_ml.dataset import all_wavelengths
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from itertools import islice
from tqdm import tqdm
from torchvision.transforms import v2
import matplotlib.pyplot as plt
from src.torch.vit.ig import single_aarp, make_predictions, do_ig
from src.torch.vit.utils import dfs_from_metadata, get_data_model, needs_resize
from src.torch.vit.train import TrainingConfig

VALID_MODEL_TYPES = {"vit": "deepflare_vit", "vit-pretrained": "vit_pretrained"}

__all__ = [
    "plot_intensity_distribution",
    "get_intensities_using_attributions",
        ]

def run_pred_and_ig(aarp_id, metadata_df, transform, model, device, multiply_by_inputs=True, stride=1, target_mode='true_label', baseline_image=None, channel_indices=None, resize_to=None):
    aarp_id_df = metadata_df.query(f'aarp_id == {aarp_id}')
    s_aarp = single_aarp(aarp_id, aarp_id_df)
    s_images = s_aarp.get_images()
    if channel_indices is not None:
        # Models trained on an AIA-channel subset: keep only those channels,
        # matching the order the model saw during training.
        s_images = s_images[:, channel_indices]
    if stride > 1:
        s_images = s_images[::stride]
    # Use the model to make predictions
    tensor_images = torch.from_numpy(s_images).to(torch.float32)  # shape: [x, 7, 512, 512]
    tensor_data = transform(tensor_images)
    if resize_to is not None:
        tensor_data = v2.Resize(resize_to)(tensor_data)
    tensor_data = tensor_data.to(device)
    dataset = TensorDataset(tensor_data)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    attributions = []
    label = s_aarp.label
    ib_size = 1
    n_images = s_images.shape[0]

    if baseline_image is not None:
        # Fixed baseline (e.g. channel-mean image), transformed once and reused for every frame.
        baseline_fixed = transform(torch.from_numpy(baseline_image).to(torch.float32).unsqueeze(0))
        if resize_to is not None:
            baseline_fixed = v2.Resize(resize_to)(baseline_fixed)
        baseline_fixed = baseline_fixed.to(device)

    with torch.no_grad():
        for (batch,) in tqdm(islice(loader, n_images), total=n_images, desc=f"IG aarp={aarp_id}"):
            batch = batch.to(device)
            if baseline_image is not None:
                baseline = baseline_fixed
            else:
                baseline = transform(torch.zeros_like(batch))
                baseline = baseline.to(device)
            ig_b0 = do_ig(batch, baseline, label=label, ib_size=ib_size, model=model,
                          multiply_by_inputs=multiply_by_inputs, target_mode=target_mode)
            attributions.append(ig_b0)
            del batch, ig_b0
            torch.cuda.empty_cache()
            gc.collect()
    del tensor_images, tensor_data, dataset
    return s_images, attributions

def get_intensities_using_attributions(attributions_list, images_list,
                                       channel, percentile_level):
    # Guard against empty lists (e.g. subset splits with no samples for a class)
    if not attributions_list or not images_list:
        return np.array([])
    attributions_comb = np.array([item[channel] for attribution in attributions_list
                                  for item in attribution])
    images_comb= np.array([item[channel] for images in images_list
                           for item in images])
    # Guard against degenerate array shape when all lists were empty iterables
    if attributions_comb.ndim < 3:
        return np.array([])
    thresholds = np.percentile(attributions_comb, percentile_level, axis=(1, 2))
    thresholds_expanded = thresholds[:, None, None]
    filtered_images = np.where(attributions_comb  > thresholds_expanded, images_comb, 0)
    return filtered_images.flatten()

def plot_intensity_distribution(images:tuple, attributions:tuple, percentile_levels, passband:int, x_range=(0,6),
                                nbins=30, alpha=0.4, figsize=(24,5), dpi=150, model_passbands=None):
    # model_passbands: passbands present in the image/attribution arrays, in order.
    # None means all 7 AIA channels; a subset (e.g. [94, 131]) changes the index mapping.
    channel = (model_passbands if model_passbands is not None else all_wavelengths).index(passband)
    images_list_neg, images_list_pos = images
    attributions_list_neg, attributions_list_pos = attributions

    fig, axes = plt.subplots(1, len(percentile_levels), figsize=figsize, dpi=dpi, sharey=True)

    for idx, percentile_level in enumerate(percentile_levels):

        flattened_int_pos = get_intensities_using_attributions(
            attributions_list_pos, images_list_pos,
            channel=channel, percentile_level=percentile_level
        )

        flattened_int_neg = get_intensities_using_attributions(
            attributions_list_neg, images_list_neg,
            channel=channel, percentile_level=percentile_level
        )

        ax = axes[idx]

        for class_idx, class_intensities in enumerate((flattened_int_neg, flattened_int_pos)):
            valid = class_intensities[class_intensities >= 1]
            if valid.size == 0:  # skip empty class (e.g. no neg samples in subset)
                continue
            ax.hist(np.log(valid), bins=nbins, range=x_range,
                density=True, label=("Flared" if class_idx == 1 else "Non-Flared"), alpha=alpha)

        ax.set_title(fr"${percentile_level}^{{\mathrm{{th}}}}$ percentile")
        fig.text(0.02, 0.5, f"Passband {passband}", rotation=90, va="center")
        if idx == 0:
            ax.set_ylabel("Density")
        ax.set_xlabel("log(Intensity)")
        ax.legend()
    return fig, axes

if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-path",      default="solar_dataset.json")
    parser.add_argument("--model-path",     default="outputs/glad-shape-197/trained_model.pth",
                        help="Path to trained ViT model checkpoint.")
    parser.add_argument("--output-neg",     default="data/intermediate-outs/attributions_neg.pt")
    parser.add_argument("--output-pos",     default="data/intermediate-outs/attributions_pos.pt")
    parser.add_argument("--stride",         type=int, default=1,
                        help="Use every Nth frame per AARP for IG (validated up to stride=4 with negligible histogram distortion).")
    parser.add_argument("--target-mode",    default="true_label", choices=["true_label", "flare", "margin"],
                        help="Which output IG attributes to: 'true_label' (this AARP's own class, default), "
                             "'flare' (always the flare logit), or 'margin' (logit_flare - logit_nonflare).")
    parser.add_argument("--stats-file",     default="stats.pkl",
                        help="Path to normalization stats pickle (e.g. stats_raw.pkl for pre-fix models).")
    parser.add_argument("--subset",         default="test", choices=["training", "validation", "test"],
                        help="Dataset split to compute attributions on (default: test).")
    parser.add_argument("--channels",       type=int, nargs="+", default=None,
                        help="AIA passbands the model was trained on, e.g. --channels 94 131. "
                             f"Choices: {all_wavelengths}. Default: all 7, in wavelength order.")
    parser.add_argument("--model-type",     default="vit", choices=list(VALID_MODEL_TYPES),
                        help="Model architecture: 'vit' (DeepFlare_ViT, default) or "
                             "'vit-pretrained' (torchvision vit_l_16).")
    parser.add_argument("--aarp-id",        type=int, nargs="+", default=None,
                        help="Restrict to specific AARP ID(s) instead of the full subset "
                             "(e.g. for a cheap smoke test before a full IG sweep -- this "
                             "script is the most expensive one in the battery).")
    args = parser.parse_args()

    channel_indices = None
    if args.channels is not None:
        unknown = [c for c in args.channels if c not in all_wavelengths]
        if unknown:
            parser.error(f"Unknown channel(s) {unknown}. Choices: {all_wavelengths}")
        channel_indices = [all_wavelengths.index(c) for c in args.channels]

    # Load Data and Model
    with open(args.json_path, 'r') as json_file:
        metadata = json.load(json_file)

    training_df, val_df, test_df = dfs_from_metadata(metadata)
    subset_df = {"training": training_df, "validation": val_df, "test": test_df}[args.subset]
    if args.aarp_id is not None:
        subset_df = subset_df[subset_df.aarp_id.isin(args.aarp_id)]

    config = TrainingConfig(json_path=args.json_path, stats_file=args.stats_file,
                            channel_indices=channel_indices,
                            model_type=VALID_MODEL_TYPES[args.model_type])
    config.trained_model_path = args.model_path
    _, model, transform, device = get_data_model(config)
    resize_to = needs_resize(config)
    print(f"Model type : {args.model_type}" + (f"  (resize to {resize_to})" if resize_to else ""))

    attributions_list_pos = []
    attributions_list_neg = []
    images_list_pos = []
    images_list_neg = []

    for aarp_id in subset_df.query('label == 1').aarp_id.unique():
        print(aarp_id)
        s_images, attributions= run_pred_and_ig(aarp_id, subset_df, transform, model, device, stride=args.stride, target_mode=args.target_mode, channel_indices=channel_indices, resize_to=resize_to)
        print(f"{s_images.shape=}")
        print(f"{len(attributions)=}")
        attributions_list_pos.append(attributions)
        del attributions
        images_list_pos.append(s_images)
        gc.collect()
        torch.cuda.empty_cache()

    for aarp_id in subset_df.query('label == 0').aarp_id.unique():
        print(aarp_id)
        s_images, attributions= run_pred_and_ig(aarp_id, subset_df, transform, model, device, stride=args.stride, target_mode=args.target_mode, channel_indices=channel_indices, resize_to=resize_to)
        print(f"{s_images.shape=}")
        print(f"{len(attributions)=}")
        attributions_list_neg.append(attributions)
        del attributions
        images_list_neg.append(s_images)
        gc.collect()
        torch.cuda.empty_cache()

    import os
    os.makedirs(os.path.dirname(args.output_neg), exist_ok=True)
    torch.save(attributions_list_neg, args.output_neg)
    torch.save(attributions_list_pos, args.output_pos)
