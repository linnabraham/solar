import sys
sys.path.append("..")
import argparse
import torch
import math
from tqdm import tqdm
import time
from torch.utils.data import Dataset, DataLoader, RandomSampler
from torchvision.transforms import v2
import pickle
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score, confusion_matrix
import numpy as np
import os
import wandb
from utils.torch_utils import global_parser, SaveBestModel
from aia_ds import aia_euv, CustomTransform

class AlexNet(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.5) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(7, 64, kernel_size=11, stride=4, padding=2),  # Change 3 to 7
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def train_loop():
    
    threshold = 5000  # GPU memory threshold measured in megabytes

    wandb_dir = wandb.run.name
    output_dir = os.path.join("output", wandb_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # initialize the callback
    save_best_model_callback = SaveBestModel(monitor='val_loss', mode='min')

    #num_samples = 150 
    num_samples = len(val_loader.dataset)
    print(f"Randomly sampling {num_samples} items from validation dataset")

    #train_loader = dummy_data(train_dataset, num_samples=num_samples, 
    #        batch_size=args.batch_size)

    #val_loader = dummy_data(validation_dataset, num_samples=num_samples, 
    #        batch_size=args.batch_size)

    print("train size", len(train_loader.dataset))
    print("validation size", len(val_loader.dataset))

    n_steps_per_epoch = math.ceil(len(train_loader.dataset) / args.batch_size)
    print(f"Steps per epoch:{n_steps_per_epoch}")

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0

        start_time = time.time()

        print(f"Epoch:{epoch}")

        for step, (inputs, labels) in tqdm(enumerate(train_loader), total=len(train_loader), leave=False):

            current_memory = torch.cuda.memory_allocated() / (1024 ** 2)
            #print(f"Allocated GPU memory ({current_memory} MB)")

            #memory_reserved = torch.cuda.memory_reserved() / (1024 ** 2)
            #print(f"GPU memory reserved: {memory_reserved}")

            if current_memory > threshold:
                print(f"GPU memory usage ({current_memory} MB) exceeds threshold. Breaking the script.")
                sys.exit(0)

            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            labels = labels.float()
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


        val_loss, accuracy, precision, recall = validate_model(model, val_loader, criterion, num_samples = num_samples, device=device)

        val_metrics = {"val/val_loss": val_loss, 
                       "val/val_accuracy": accuracy,
                       "val/precision":precision,
                       "val/recall":recall}

        print(val_metrics)
        wandb.log({**metrics, **val_metrics})

        # run the callback to save the model
        save_best_model_callback(val_loss, model, os.path.join(output_dir,"trained_model.pth"))
        epoch_loss = running_loss / len(train_dataset)
        print(f"Epoch loss: {epoch_loss}")

        epoch_time = time.time() - start_time
        print(f"Time taken to run single epoch: {epoch_time/60} mins")

        max_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # Convert to megabytes
        print(f"Maximum GPU memory usage: {max_memory} MB")

        max_memory_reserved = torch.cuda.max_memory_reserved() / (1024 ** 2)
        print(f"Maximum GPU memory reserved: {max_memory_reserved}")

def validate_model(model, val_dl, loss_func, num_samples, device, threshold=0.5):
    model.eval()
    val_loss = 0.
    correct = 0
    precision = []
    recall = []
    total_tn = 0
    total_fp = 0
    total_fn = 0
    total_tp = 0
    with torch.inference_mode():
        for i, (images, labels) in tqdm(enumerate(val_dl), total=len(val_dl), leave=False):
            images, labels = images.to(device), labels.to(device)

            # Forward pass ➡
            outputs = model(images).squeeze()
            labels = labels.float()
            val_loss += loss_func(outputs, labels).item()*labels.size(0)
            # Compute accuracy and accumulate
            pred_scores = outputs.data
            predicted = (pred_scores > threshold).int()
            correct += (predicted == labels).sum().item()

            #pred_scores, predicted = torch.max(outputs.data, 1)

            precision.append(precision_score(labels.cpu(), predicted.cpu(), zero_division=0))
            recall.append(recall_score(labels.cpu(), predicted.cpu(), zero_division=0))
            tn, fp, fn, tp  = confusion_matrix(labels.cpu(), predicted.cpu(), labels=[0,1]).ravel()
            total_tn += tn
            total_fp += fp
            total_fn += fn
            total_tp += tp

    print("Total TN:", total_tn)
    print("Total FP:", total_fp)
    print("Total FN:", total_fn)
    print("Total TP:", total_tp)

    return val_loss/num_samples, correct/num_samples, np.mean(np.array(precision)), np.mean(np.array(recall))

def dummy_data(dataset, num_samples, batch_size):
    """
    Create a small dataset for testing by sampling from our actual dataset
    """
    sampler = RandomSampler(dataset, num_samples=num_samples)
    dl_loader  = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    return dl_loader

if __name__=="__main__":

    parser = global_parser()
    parser.add_argument("-batch-size", "--batch-size", type=int, default=32)
    parser.add_argument("-epochs", "--epochs", type=int, default=5)
    parser.add_argument('-lr', '--lr', type=float, default=0.001)
    args = parser.parse_args()

    args_dict = vars(args)
    print("Args dict:", args_dict)

    model = AlexNet(num_classes=1)

    project_name = "flare_torch"

    wandb.init(
        project= project_name,
        config= args_dict
            )

    with open('../vit/stats.pkl', 'rb') as f:
        stats = pickle.load(f)

    means = [stats['mean'][f'channel_{i}'] for i in range(7)]
    stds = [stats['std'][f'channel_{i}'] for i in range(7)]

    train_dataset = aia_euv(args.json_path, subset='training', transform=v2.Compose([
        CustomTransform(means, stds, zscore=True),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomVerticalFlip(p=0.5)
        ]))

    validation_dataset = aia_euv(args.json_path, subset='validation', transform=v2.Compose([CustomTransform(means, stds, zscore=True)]))

    print("train size", len(train_dataset))
    print("validation size", len(validation_dataset))

    train_loader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle=True)
    val_loader = DataLoader(validation_dataset, batch_size = args.batch_size, shuffle=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}")

    torch.cuda.reset_peak_memory_stats()

    model.to(device)
    train_loop()
