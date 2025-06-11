import json
import pandas as pd
from aarp_ml.dataset import all_wavelengths
import torch
from aarp_ml.torch.trained_model import trained_model
from aarp_ml.torch.dataset import AIALogTransform
import pickle
import numpy as np
from astro_utils.utils import read_fits_single
from torch.utils.data import DataLoader, TensorDataset
from captum.attr import IntegratedGradients
from captum.attr._utils.visualization import _normalize_attr
from aarp_ml.torch.model import DeepFlare_ViT
from itertools import islice
import matplotlib.pyplot as plt
import math
import sys

def plot_images(images, cols=5, cmap='viridis', gap=0, dpi=100, show=False):
    """
    Plots a compact grid of images using fig.add_axes, returning the figure.

    Parameters:
    - images: np.ndarray of shape [n, H, W]
    - cols: Number of columns
    - cmap: Colormap
    - gap: Space (in pixels) between images
    - dpi: Dots per inch for sizing

    Returns:
    - fig: Matplotlib Figure object (not shown by default)
    """
    n, H, W = images.shape
    rows = math.ceil(n / cols)

    # Compute figure size in inches
    total_width_px = cols * W + (cols - 1) * gap
    total_height_px = rows * H + (rows - 1) * gap
    figsize = (total_width_px / dpi, total_height_px / dpi)

    fig = plt.figure(figsize=figsize, dpi=dpi)

    for idx in range(n):
        row = idx // cols
        col = idx % cols

        # Position in figure coordinates [0,1]
        left = (col * (W + gap)) / total_width_px
        bottom = 1 - ((row + 1) * H + row * gap) / total_height_px
        width = W / total_width_px
        height = H / total_height_px

        ax = fig.add_axes([left, bottom, width, height])
        ax.imshow(images[idx], cmap=cmap)
        ax.axis('off')
    if show:
        plt.show()
    else:
        plt.close(fig)
        return fig

class single_aarp:
    def __init__(self,aarp_id, dataframe):
        self.aarp_id = aarp_id
        self.dataframe = dataframe

    def get_images(self):
        all_filepaths = self.dataframe[all_wavelengths].values.tolist()
        images = np.zeros(shape=(len(all_filepaths), 7, 512, 512))
        ob_idx = 0
        for filepaths in all_filepaths:
            img_list = []
            for i, pb in zip(filepaths, all_wavelengths) :
                try:
                    data_fits = read_fits_single(i)
                except:
                    continue
                else:
                    img_list.append(data_fits)
                    img_arr = np.array(img_list)
            images[ob_idx,::] = img_arr
        ob_idx +=1
        return images

def do_ig(features, label, ib_size=1):
    ig = IntegratedGradients(model)
    baseline_zero = torch.zeros_like(features)
    labels  = torch.tensor(1, dtype=torch.int32)
    ig_b0, _ = ig.attribute(features, baseline_zero, target=labels, n_steps=100, internal_batch_size=ib_size,
                                        return_convergence_delta=True)
    return ig_b0

def make_predictions(dataset):
    model.eval()
    loader = DataLoader(dataset, batch_size=16, shuffle=False)

    results = []

    with torch.no_grad():
        for (batch,) in loader:  # each batch is a 1-tuple from TensorDataset
            batch = batch.to(device)
            output = model(batch)
            results.append(output.cpu())

    predictions = torch.cat(results, dim=0)
    print(predictions)

    probs = torch.softmax(predictions, dim=1)
    predicted_scores, predicted_labels = torch.max(predictions, dim=1)
    return probs, predicted_scores, predicted_labels

def ig_on_aarp_seq(aarp_id, test_df, transform, model, device):
    s_aarp = single_aarp(aarp_id, test_df.query(f'aarp_id == {aarp_id}'))
    s_images = s_aarp.get_images()

    tensor_images = torch.from_numpy(s_images).to(torch.float32)  # shape: [x, 7, 512, 512]
    tensor_data = transform(tensor_images)
    tensor_data = tensor_data.to(device)
    model = model.to(device)

    dataset = TensorDataset(tensor_data)

    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    attributions = []
    with torch.no_grad():
        for (batch,) in islice(loader, 5):  # take only first 5 batches
            batch = batch.to(device)
            ig_b0 = do_ig(batch, label=1, ib_size=1)
            attributions.append(ig_b0[0])
    return s_images, attributions

if __name__ == "__main__":
    trained_model_path = sys.argv[1]
    with open('solar_dataset.json', 'r') as json_file:
        metadata = json.load(json_file)

    training_df = pd.DataFrame(metadata['training'])
    val_df = pd.DataFrame(metadata['validation'])
    test_df = pd.DataFrame(metadata['test'])

    training_df = training_df.rename({str(i):all_wavelengths[i] for i in range(7)}, axis=1)
    val_df = val_df.rename({str(i):all_wavelengths[i] for i in range(7)}, axis=1)
    test_df = test_df.rename({str(i):all_wavelengths[i] for i in range(7)}, axis=1)

    with open('stats.pkl', 'rb') as pickle_file:
        stats_data = pickle.load(pickle_file)

    means = [stats_data.get('mean').get(f'channel_{i}') for i in range(7)]
    stds = [stats_data.get('std').get(f'channel_{i}') for i in range(7)]

    model = DeepFlare_ViT(height=512, n_classes=2, n_passbands=7).model

    transform = AIALogTransform(means=means, stds=stds)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(trained_model_path, map_location=device)
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

    aarp_ids = [377,  401, 1449, 4920]

    for aarp_id in aarp_ids:
        s_images, attributions = ig_on_aarp_seq(aarp_id, test_df, transform, model, device)
        print(s_images.shape)
        print(attributions[0].min())
        print(attributions[0].max())
        print(attributions[1].min())
        print(attributions[1].max())

