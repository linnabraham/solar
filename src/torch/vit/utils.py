import json
import torch
import gc
import numpy as np
import pickle
import pandas as pd
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
    means = [stats_data.get('mean').get(f'channel_{i}') for i in range(config.n_channels)]
    stds = [stats_data.get('std').get(f'channel_{i}') for i in range(config.n_channels)]

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
    training_df['timestamp'] = training_df['timestamp'].apply(pd.to_datetime).dt.tz_localize(None)
    val_df['timestamp'] = val_df['timestamp'].apply(pd.to_datetime).dt.tz_localize(None)
    test_df['timestamp'] = test_df['timestamp'].apply(pd.to_datetime).dt.tz_localize(None)
    training_df = training_df.rename({str(i):all_wavelengths[i] for i in range(7)}, axis=1)
    val_df = val_df.rename({str(i):all_wavelengths[i] for i in range(7)}, axis=1)
    test_df = test_df.rename({str(i):all_wavelengths[i] for i in range(7)}, axis=1)
    return training_df, val_df, test_df
