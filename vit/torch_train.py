import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from base import BaseModel
from vit_pytorch import ViT
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch.utils.data import Dataset, DataLoader
import json
from astropy.io import fits
import time
import argparse
import os,sys
import wandb
import math

class SaveBestModel:
    def __init__(self, monitor='val_loss', mode='min'):
        self.monitor = monitor
        self.mode = mode
        if mode == 'min':
            self.best_value = float('inf')
            self.monitor_op = lambda x, y: x < y
        else:
            self.best_value = float('-inf')
            self.monitor_op = lambda x, y: x > y
        
    def __call__(self, val_metric, model, filepath):
        if self.monitor_op(val_metric, self.best_value):
            print(f"Validation {self.monitor}: {val_metric} improved from {self.best_value} to {val_metric}. Saving model...")
            self.best_value = val_metric
            torch.save(model.state_dict(), filepath)
        else:
            print(f"Validation {self.monitor}: {val_metric} did not improve from {self.best_value}.")

class aia_euv(Dataset):
    def __init__(self, json_path, subset):
        self.data = self._load_data(json_path, subset)

    def _load_data(self, json_file, subset):
        with open(json_file, 'r') as f:
            data_dict = json.load(f)
        return data_dict.get(subset, [])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        features = [self._read_fits_file(item[str(i)]) for i in range(7)]  # Read FITS files
        features = np.stack(features, axis=0)  # Stack along a new axis
        label = item['label']
        features = torch.tensor(features, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)
        return features, label

    def _read_fits_file(self, file_path):
        with fits.open(file_path) as hdul:
            data = hdul[0].data
        return data

class DeepFlare_ViT(BaseModel):
    def __init__(self,**kwargs):
        super(DeepFlare_ViT,self).__init__(**kwargs)
        n_passbands = kwargs.pop('n_passbands',None)
        height = kwargs.pop('height',None)
        n_classes = kwargs.pop('n_classes',None)
        dropout_prob = kwargs.pop('dropout',0.3)
        if n_passbands is None or height is None or n_classes is None:
            raise ValueError("Number of input passbands (filters) must be given!")
        self.model = ViT(image_size=height,patch_size=64,num_classes=n_classes,
                         dim=1024,depth=4,heads=16,channels=n_passbands,mlp_dim=9,
                         dropout=dropout_prob,emb_dropout=dropout_prob,pool="mean")
def train_loop():
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    threshold = 5000  # GPU memory threshold measured in megabytes

    n_steps_per_epoch = math.ceil(len(train_loader.dataset) / args.batch_size)
    print(f"Steps per epoch:{n_steps_per_epoch}")

    wandb_dir = wandb.run.name
    output_dir = os.path.join("output", wandb_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    save_best_model_callback = SaveBestModel(monitor='val_loss', mode='min')

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0

        start_time = time.time()

        print(f"Epoch:{epoch}")

        for step, (inputs, labels) in enumerate(train_loader):

            current_memory = torch.cuda.memory_allocated() / (1024 ** 2)
            #print(f"Allocated GPU memory ({current_memory} MB)")

            #memory_reserved = torch.cuda.memory_reserved() / (1024 ** 2)
            #print(f"GPU memory reserved: {memory_reserved}")

            if current_memory > threshold:
                print(f"GPU memory usage ({current_memory} MB) exceeds threshold. Breaking the script.")
                sys.exit(0)

            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            metrics = {"train/train_loss": loss,
                       "train/epoch": (step + 1 + (n_steps_per_epoch * epoch)) / n_steps_per_epoch
                       }
            if step + 1 < n_steps_per_epoch:
                # Log train metrics to wandb 
                wandb.log(metrics)
        val_loss, accuracy, precision, recall = validate_model(model, val_loader, criterion)

        val_metrics = {"val/val_loss": val_loss, 
                       "val/val_accuracy": accuracy,
                       "val/precision":precision,
                       "val/recall":recall}

        wandb.log({**metrics, **val_metrics})

        save_best_model_callback(val_loss, model, os.path.join(output_dir,"trained_model.pth"))

        epoch_loss = running_loss / len(train_dataset)
        print(f"Epoch loss: {epoch_loss}")
        
        epoch_time = time.time() - start_time
        print(f"Time taken to run single epoch: {epoch_time/60} mins")

        max_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # Convert to megabytes
        print(f"Maximum GPU memory usage: {max_memory} MB")

        max_memory_reserved = torch.cuda.max_memory_reserved() / (1024 ** 2)
        print(f"Maximum GPU memory reserved: {max_memory_reserved}")

def validate_model(model, val_dl, loss_func):
    model.eval()
    val_loss = 0.
    with torch.inference_mode():
        correct = 0
        TP = 0
        FP = 0
        TN = 0
        FN = 0
        for i, (images, labels) in enumerate(val_dl):
            images, labels = images.to(device), labels.to(device)

            # Forward pass ➡
            outputs = model(images)
            val_loss += loss_func(outputs, labels)*labels.size(0)
            # Compute accuracy and accumulate
            pred_scores, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()

            for pred,label in zip(predicted, labels):
                if pred == 1 and label == 1:
                    TP += 1
                elif pred == 1 and label == 0:
                    FP += 1
                elif pred == 0 and label == 0:
                    TN += 1
                elif pred == 0 and label == 1:
                    FN += 1

        print("True Positives:", TP)
        print("False Positives:", FP)
        print("True Negatives:", TN)
        print("False Negatives:", FN)
        try:
            precision = TP/(TP+FP)
        except:
            precision = 0
        try:
            recall = TP/(TP+FN)
        except:
            recall = 0
    return val_loss / len(val_dl.dataset), correct / len(val_dl.dataset), precision, recall

if __name__== "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-batch-size", "--batch-size", type=int, default=32)
    parser.add_argument("-epochs", "--epochs", type=int, default=5)
    parser.add_argument('-lr', '--lr', type=float, default=0.001)
    args = parser.parse_args()

    args_dict = vars(args)

    project_name = "flare_torch"

    wandb.init(
        project= project_name,
        config= args_dict
            )

    vit_model = DeepFlare_ViT(height=512, n_classes=2, n_passbands=7)
    model = vit_model.model

    train_dataset = aia_euv('../solar_dataset.json', subset='training')
    validation_dataset = aia_euv('../solar_dataset.json', subset='validation')
    test_dataset = aia_euv('../solar_dataset.json', subset='test')

    print("Checking data specifications")
    for i in range(len(train_dataset)):
        features, label = train_dataset[i]
        print(features.shape)
        break

    print("train size", len(train_dataset))
    print("validation size", len(validation_dataset))
    print("test size", len(test_dataset))

    train_loader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle=True)
    val_loader = DataLoader(validation_dataset, batch_size = args.batch_size, shuffle=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}")

    torch.cuda.reset_peak_memory_stats()

    model.to(device)

    train_loop()
