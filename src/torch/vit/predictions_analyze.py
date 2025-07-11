import json
import pandas as pd
from aarp_ml.dataset import all_wavelengths
import torch
from aarp_ml.torch.dataset import AIALogTransform
import pickle
import numpy as np
from torch.utils.data import TensorDataset
from aarp_ml.torch.model import DeepFlare_ViT
import matplotlib.pyplot as plt
import os
from src.torch.vit.ig import single_aarp, make_predictions
import aarp_ml
from astro_utils.utils import get_start_and_end_time
import matplotlib.dates as mdates
from sunpy.timeseries import XRSTimeSeries
import warnings
import gc
from src.data_single import DatasetPaths
from src.torch.vit.utils import get_data_model, dfs_from_metadata
from src.torch.vit.train import TrainingConfig

def plot_goes(goes_ts, columns=None, xlimits=None, ax=None, figsize=(10,6), dpi=150, **kwargs):

    """
    My custom function to plot GOES data instead of using the XRSTimeseries class.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        fig = ax.figure
    plot_settings = {"xrsa": ["blue", r"0.5$-$4.0 $\mathrm{\AA}$"], "xrsb": ["red", r"1.0$-$8.0 $\mathrm{\AA}$"]}
    if columns is None:
        columns = ["xrsa", "xrsb"]
    if xlimits:
        # Convert both ends of xlimits to timezone-aware timestamps
        # xlimits = [pd.Timestamp(t).tz_localize('UTC') if pd.Timestamp(t).tzinfo is None else pd.Timestamp(t).tz_convert('UTC') for t in xlimits]
        a, b = xlimits
        data = goes_ts.truncate(a,b).data
    else:
        data = goes_ts.data
    for channel in columns:
        ax.plot(
            data.index, data[channel], "-", label=plot_settings[channel][1], color=plot_settings[channel][0], lw=1, **kwargs
        )
    ax.set_yscale("log")
    ax.set_ylim(1e-9, 1e-2)
    ax.set_ylabel("Watts m$^{-2}$")

    locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

    ax.tick_params(axis='x', rotation=45)
    labels = ['A', 'B', 'C', 'M', 'X']
    centers = np.logspace(-7.5, -3.5, len(labels))
    centers = np.logspace(-7.5, -3.5, len(labels))

    for value, label in zip(centers, labels):
        ax.text(1.02, value, label, transform=ax.get_yaxis_transform(), horizontalalignment='center')
    ax.yaxis.grid(True, "major")
    ax.xaxis.grid(False, "major")
    ax.legend()
    return fig, ax  

def plot_custom_goes_with_aarp_sampling(goes_ts, timestamps, columns=None, xlimits=None, ax=None, figsize=(10,6), dpi=150, **kwargs):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        fig = ax.figure

    fig, ax = plot_goes(goes_ts, columns=columns, xlimits=xlimits, ax=ax, figsize=figsize, dpi=dpi, **kwargs)
    if xlimits:
        xlim_start = pd.Timestamp(xlimits[0], tz="UTC")
        xlim_end = pd.Timestamp(xlimits[1], tz="UTC")
        timestamps = timestamps[timestamps.between(xlim_start, xlim_end)]
    for ts in timestamps:
        ax.axvline(ts, color='grey', linestyle='--')
    return fig, ax

def vizualize_goes_ts_predictions(goes_ts, s_aarp:single_aarp, predicted_scores, goes_event_list_df=None,
    flare_start=False, columns=["xrsb"], xlimits=None, resample=False, figsize=(10,6), dpi=150, **kwargs):

    """
    Function to visualize the model predictions on top of the GOES timeseries.
    Optionally also mark the start time of the flare in the GOES timeseries.
    """

    if resample:
        goes_ts = XRSTimeSeries(data=goes_ts.data.reindex(pd.DatetimeIndex(s_aarp.timestamps), method="nearest", tolerance=pd.Timedelta(seconds=2)), meta=goes_ts.meta)

    fig, ax = plot_goes(goes_ts, columns=['xrsb'], xlimits=xlimits, **kwargs)

    if flare_start:
        if goes_event_list_df is None:
            raise ValueError("goes_event_list should be given if flare_start is True")
        result_event_list = goes_event_list_df.query(f"harpnum == {s_aarp.aarp_id}")
        if len(result_event_list) != 1:
            warnings.warn(f"Expected 1 event, got {len(result_event_list)}")
        f_st = result_event_list[['start_time', 'peak_time', 'end_time']].values[0][0]
        ax.axvline(f_st, color='black', linestyle='--', label='flare start')
        ax.legend()

    # xlim_start = pd.Timestamp(xlimits[0], tz="UTC")
    # xlim_end = pd.Timestamp(xlimits[1], tz="UTC")
    xlim_start = pd.Timestamp(xlimits[0])
    xlim_end = pd.Timestamp(xlimits[1])
    mask = s_aarp.timestamps.between(xlim_start, xlim_end)

    # If predicted_scores is a NumPy array:
    sliced_scores = predicted_scores[mask.to_numpy()]
    filtered_ts = s_aarp.timestamps[s_aarp.timestamps.between(xlim_start, xlim_end)]
    ax2 = ax.twinx()
    # ax2.scatter(s_aarp.timestamps.tolist(), predicted_scores.numpy(), color='brown', linestyle='-', linewidth=1, s=1, label='Predicted Scores')'
    ax2.scatter(filtered_ts, sliced_scores, color='blue', linestyle='-', linewidth=1, s=1, label='Predicted Scores')
    ax2.tick_params(axis='y', pad=15)  
    ax2.set_ylim(-0.1, 1.1)
    # ax.plot(filtered_ts, sliced_scores)
    ax2.legend()
    plt.title(f"ViT prediction scores for AARP {s_aarp.aarp_id} overlaid on GOES X-ray timeseries", fontsize=9)
    return fig, ax

def viz_predictions_2(goes_ts, s_aarp:single_aarp, predicted_scores, flare_start=None, columns=["xrsb"],
    xlimits=None,figsize=(10,6), dpi=150):
    """
    Another way to visualize the same as above
    """
    t_stamps = s_aarp.timestamps.tolist()

    fig, ax = plot_custom_goes_with_aarp_sampling(goes_ts, t_stamps, columns=["xrsb"], xlimits=xlimits, figsize=(10,6))

    if flare_start:
        ax.axvline(flare_start, color='black', linestyle='--', label='flare start')

    xlim_start = pd.Timestamp(xlimits[0], tz="UTC")
    xlim_end = pd.Timestamp(xlimits[1], tz="UTC")
    mask = s_aarp.timestamps.between(xlim_start, xlim_end)

    # If predicted_scores is a NumPy array:
    sliced_scores = predicted_scores[mask.to_numpy()]
    filtered_ts = s_aarp.timestamps[s_aarp.timestamps.between(xlim_start, xlim_end)]

    visible_indices = [i for i, ts in enumerate(filtered_ts) if xlim_start <= ts <= xlim_end]
    for i in visible_indices:
        ts = filtered_ts.iloc[i]
        pred_score = predicted_scores[i]

        if xlim_start <= ts <= xlim_end:
            y_base = ax.get_ylim()[1]
            y_offset_factor = (i % 4) * 0.05  # 4 levels
            y_pos = y_base * (0.35 - y_offset_factor)
            ax.annotate(f"{pred_score:.2f}",
                        xy=(ts, y_pos),
                        xytext=(0, 2),  # small offset in points
                        textcoords='offset points',
                        fontsize=10,
                        fontweight='bold',
                        rotation=90,
                        color='blue',
                        ha='center',
                        # va='bottom',
                        annotation_clip=True,
                        bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=0.5))

    ax.legend()
    plt.title(f"ViT prediction scores for AARP {s_aarp.aarp_id} overlaid on GOES X-ray timeseries", fontsize=9)
    plt.show()

def get_aarp_seq_dataset(s_aarp, transform, device):
    s_images = s_aarp.get_images()
    # plot_image_grid(s_images[3], show=True)
    # plt.close()

    # Use the model to make predictions
    tensor_images = torch.from_numpy(s_images).to(torch.float32)  # shape: [x, 7, 512, 512]
    tensor_data = transform(tensor_images)
    tensor_data = tensor_data.to(device)
    dataset = TensorDataset(tensor_data)
    return dataset

def make_prediction_plot(aarp_id, metadata_df, transform, model, device, output_home):
    output_dir = f"{output_home}/{aarp_id}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        print(f"Output directory {output_dir} already exists, skipping {aarp_id}")
        return None

    print(f"Using {aarp_id=}")

    aarp_id_df = metadata_df.query(f'aarp_id == {aarp_id}')
    s_aarp = single_aarp(aarp_id, aarp_id_df)

    dataset = get_aarp_seq_dataset(s_aarp, transform, device)
    predictions = make_predictions(dataset, model=model, device=device)
    torch.cuda.empty_cache()
    gc.collect()
    goes_event_list_path = DatasetPaths(parent_dir=".").goes_event_with_aarp
    print(f"{goes_event_list_path}")
    goes_event_list = pd.read_csv(goes_event_list_path, parse_dates=["event_date", "start_time", "peak_time","end_time"])

    softmax_predictions = torch.softmax(predictions, dim=1)
    predicted_scores, predicted_labels = torch.max(softmax_predictions , dim=1)

    # plt.plot(predicted_scores)
    # plt.savefig(f"{output_dir}/predicted_scores.png", bbox_inches="tight", dpi=150)
    # plt.close()

    # Make plots combining data with predictions
    print("Half width duration of observations", (s_aarp.timestamps.max() - s_aarp.timestamps.min())/2)
    fl_start, fl_end = get_start_and_end_time(s_aarp.get_midtime(), 24*4.5*60)
    goes_ts = aarp_ml.utils.run_fetch_goes(fl_start, fl_end)

    matched_events = goes_event_list.query(f"harpnum == {aarp_id}")
    if not matched_events.empty:
        f_st = matched_events[['start_time', 'peak_time', 'end_time']].values[0][0]
        print("Start of the flare", f_st)
        flare_start = True
    else:
        flare_start = False

    # Plot the GOES data with predictions overlaid on top using a separate y-axis
    # Also mark the flare start time
    xlimits = (s_aarp.timestamps.min(), s_aarp.timestamps.max())
    fig, ax = vizualize_goes_ts_predictions(goes_ts, s_aarp, predicted_scores, goes_event_list_df=goes_event_list,
                                            flare_start=flare_start, xlimits=xlimits, resample=False, figsize=(6,4), alpha=0.4)
    fig.savefig(f"{output_dir}/goes_with_predictions.png", bbox_inches="tight", dpi=150)
    plt.close(fig)

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
        make_prediction_plot(aarp_id, test_df, transform, model, device, output_home)
    for aarp_id in val_df.aarp_id.unique().tolist():
        make_prediction_plot(aarp_id, val_df, transform, model, device, output_home)

if __name__=="__main__":
    main()

