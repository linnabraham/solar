import sys
sys.path.append("..")
import matplotlib.pyplot as plt
import numpy as np
import pickle
from torchvision.transforms import v2
from torch.utils.data import DataLoader
import matplotlib
import sunpy.visualization.colormaps as cm
from utils.torch_utils import global_parser
import itertools
import torch
torch.manual_seed(42)
from aia_ds import aia_euv, CustomTransform
from train import prepare_dataset

aia_cmaps = {   0:'sdoaia94',
                1:'sdoaia131',
                2:'sdoaia171',
                3:'sdoaia193',
                4:'sdoaia211',
                5:'sdoaia304',
                6:'sdoaia335'}

def plot_7ch_images(image):
    image = image.numpy()
    image = np.moveaxis(image,0,2)
    plt.figure(figsize=(12,6))
    for i in range(image.shape[2]):
        cmap = matplotlib.colormaps[aia_cmaps[i]]
        plt.subplot(2,4,i+1)
        ax = plt.gca()
        im = ax.imshow(image[:,:,i], origin="lower", cmap=cmap)
        plt.axis("off")
        plt.title(aia_cmaps[i])
        plt.colorbar(im, shrink=0.8)

def plot_7ch_images_mixed(image):
    image = image.numpy()
    image = np.moveaxis(image,0,2)
    plt.figure(figsize=(12,6))
    for i in range(image.shape[2]):
        ch_idx = [2,5,0,1,3,4,6]
        cmap = matplotlib.colormaps[aia_cmaps[ch_idx[i]]]
        plt.subplot(2,4,i+1)
        ax = plt.gca()
        im = ax.imshow(image[:,:,i], origin="lower", cmap=cmap)
        plt.axis("off")
        plt.title(aia_cmaps[ch_idx[i]])
        plt.colorbar(im, shrink=0.8)

def get_idx_from_json(json_data, target_label):
    for idx, item in enumerate(json_data):
        if item['label'] == target_label :
            return idx 

def get_idx_from_dl(dataloader, target_label):
    for data in dataloader:
        images, labels , idx = data
        if labels == target_label:
            return images, labels, idx

#def plot_image(images):
if __name__=="__main__": 
    parser = global_parser()
    parser.add_argument("-batch-size", "--batch-size", type=int, default=32)
    parser.add_argument("-json-path", "--json-path")
    parser.add_argument("-epochs", "--epochs", type=int, default=5)
    parser.add_argument('-lr', '--lr', type=float, default=0.001)
    parser.add_argument("-dummy-data", "--dummy-data",action='store_true', help="Train first on a small dataset sampled from the original")
    parser.add_argument("-num-samples", "--num-samples", default=150, help="""number of samples to take from original data;
            only valid if --dummy-data set to True""")

    args = parser.parse_args()
    config = vars(args)
    config['json_path'] = "../solar_dataset.json"

    #train_ds, val_ds = prepare_dataset(args)

    train_ds = aia_euv(args.json_path, subset='training')
    json_data = train_ds.data
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)

    images, labels, idx = get_idx_from_dl(train_loader, target_label=1)
    images = images.squeeze()
    plot_7ch_images(images)
    plt.savefig("data_E5_pos.png", bbox_inches="tight")
    print("Dataset index", idx.item())
    print("AARP ID", json_data[idx.item()]['aarp_id'])
    print("Timestamp", json_data[idx.item()]['timestamp'])
    print("Label", json_data[idx.item()]['label'])
    

    images, labels, idx = get_idx_from_dl(train_loader, target_label=0)
    images = images.squeeze()
    plot_7ch_images(images)
    plt.savefig("data_E5_neg.png", bbox_inches="tight")
    print("Dataset index", idx.item())
    print("AARP ID", json_data[idx.item()]['aarp_id'])
    print("Timestamp", json_data[idx.item()]['timestamp'])
    print("Label", json_data[idx.item()]['label'])

    with open('../vit/stats.pkl', 'rb') as f:
        stats = pickle.load(f)

    means = [stats['mean'][f'channel_{i}'] for i in [2,5,0,1,3,4,6]]
    stds = [stats['std'][f'channel_{i}'] for i in [2,5,0,1,3,4,6]]

    
    config['json_path'] = "solar_dataset.json"
    train_ds = aia_euv(args.json_path, subset='training', transform=v2.Compose([
        CustomTransform(means, stds, zscore=True)
        ]))

    json_data = train_ds.data
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)

    images, labels, idx = get_idx_from_dl(train_loader, target_label=1)
    images = images.squeeze()
    plot_7ch_images_mixed(images)
    plt.savefig("data_E6_pos.png", bbox_inches="tight")
    print("Dataset index", idx.item())
    print("AARP ID", json_data[idx.item()]['aarp_id'])
    print("Timestamp", json_data[idx.item()]['timestamp'])
    print("Label", json_data[idx.item()]['label'])
    

    images, labels, idx = get_idx_from_dl(train_loader, target_label=0)
    images = images.squeeze()
    plot_7ch_images_mixed(images)
    plt.savefig("data_E6_neg.png", bbox_inches="tight")
    print("Dataset index", idx.item())
    print("AARP ID", json_data[idx.item()]['aarp_id'])
    print("Timestamp", json_data[idx.item()]['timestamp'])
    print("Label", json_data[idx.item()]['label'])

