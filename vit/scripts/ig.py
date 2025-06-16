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
import sys

class single_aarp:
    def __init__(self,aarp_id, dataframe):
        self.aarp_id = aarp_id
        self.dataframe = dataframe

    @property
    def timestamps(self):
        return self.dataframe.timestamp

    def get_midtime(self):
        return self.timestamps.iloc[len(self.dataframe)//2]

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
            if img_list:
                img_arr = np.array(img_list)
                images[ob_idx] = img_arr
                ob_idx +=1
        return images

def do_ig(features, label, ib_size=1, model=None):
    ig = IntegratedGradients(model)
    baseline_zero = torch.zeros_like(features)
    labels  = torch.tensor(1, dtype=torch.int32)
    ig_b0, _ = ig.attribute(features, baseline_zero, target=labels, n_steps=100, internal_batch_size=ib_size,
                                        return_convergence_delta=True)
    return ig_b0

def make_predictions(dataset, batch_size=16, model=None, device=None, probabilities=False):
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    results = []

    with torch.no_grad():
        for (batch,) in loader:  # each batch is a 1-tuple from TensorDataset
            batch = batch.to(device)
            output = model(batch)
            results.append(output.cpu())

    predictions = torch.cat(results, dim=0)

    if probabilities:
        return torch.softmax(predictions, dim=1)
    else:
        return predictions

def ig_on_aarp_seq(aarp_id, test_df, transform, device, model, label, ib_size, n_images=None):
    s_aarp = single_aarp(aarp_id, test_df.query(f'aarp_id == {aarp_id}'))
    s_images = s_aarp.get_images()
    
    if n_images is None:
        n_images = s_images.shape[0]
        print(f"{n_images=}")
        
    tensor_images = torch.from_numpy(s_images).to(torch.float32)  # shape: [x, 7, 512, 512]
    tensor_data = transform(tensor_images)
    tensor_data = tensor_data.to(device)
    model = model.to(device)

    dataset = TensorDataset(tensor_data)
    
    #TODO: Experiment with a batched implementation of IG
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    attributions = []
    with torch.no_grad():
        for (batch,) in islice(loader, n_images):
            batch = batch.to(device)
            ig_b0 = do_ig(batch, label=label, ib_size=ib_size, model=model)
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
        print(f"{aarp_id=}")
        t_stamps = test_df.query(f'aarp_id == {aarp_id}').timestamp
        print(f"{min(t_stamps)=}, {max(t_stamps)=}")
        s_aarp = single_aarp(aarp_id, test_df.query(f'aarp_id == {aarp_id}'))
        s_images = s_aarp.get_images()

        tensor_images = torch.from_numpy(s_images).to(torch.float32)  # shape: [x, 7, 512, 512]
        tensor_data = transform(tensor_images)
        tensor_data = tensor_data.to(device)
        model = model.to(device)
        dataset = TensorDataset(tensor_data)

        predictions = make_predictions(dataset, model=model, device=device, probabilities=True)
        predicted_scores, predicted_labels = torch.max(predictions, dim=1)

        print(predicted_labels)
        for pred_label, t_stamp in zip(predicted_labels, t_stamps):
            print(pred_label, t_stamp)
