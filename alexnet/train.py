import sys
sys.path.append("..")
import os
import argparse
import torch
import math
from torch.utils.data import Dataset, DataLoader, RandomSampler
from torchvision.transforms import v2
import pickle
import numpy as np
import wandb
import time
from tqdm import tqdm
from utils.torch_utils import global_parser
from aia_ds import aia_euv, CustomTransform
from torch_alexnet import AlexNet

def train_one_epoch(model, train_loader, loss_fn, optimizer, epoch, n_steps_per_epoch):
    running_loss = 0.
    example_ct = 0

    for step, data_batch in tqdm(enumerate(train_loader), total=len(train_loader), leave=False):

        inputs, labels = data_batch
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs).squeeze()
        labels = labels.float()
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

        example_ct += inputs.size(0)

        metrics = {"train/train_loss": loss.item(),
                   "train/epoch": (epoch + (step + 1) / n_steps_per_epoch),
                    "train/example_ct": example_ct
                   }

        if step +1 < n_steps_per_epoch:
            # Log train metrics to wandb
            wandb.log(metrics)
            #print(metrics)
    return running_loss/example_ct

def train_loop(model, train_loader, val_loader, n_steps_per_epoch, output_dir, args, device):
    loss_fn = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_vloss = 1_000_000.

    for epoch in range(args.epochs):
        #print(f"Epoch:{epoch+1}")

        model.train(True)
        avg_train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, epoch, n_steps_per_epoch)

        running_vloss = 0.0
        model.eval()
        with torch.no_grad():
            for i, vdata in tqdm(enumerate(val_loader), total=len(val_loader), leave=False):
                vinputs, vlabels = vdata
                vinputs, vlabels = vinputs.to(device), vlabels.to(device)
                voutputs = model(vinputs).squeeze()
                vlabels = vlabels.float()
                vloss = loss_fn(voutputs, vlabels)
                running_vloss += vloss

        avg_vloss = running_vloss/(i+1)
        epoch_metrics = {"train/epoch":epoch+1,
                         "train/avg_train_loss": avg_train_loss,
                         "val/val_loss":avg_vloss.item()
                         }
        wandb.log({**epoch_metrics})
        #print("Epoch metrics", epoch_metrics)

        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = os.path.join(output_dir,"trained_model.pth" )
            torch.save(model.state_dict(), model_path)

def prepare_dataset(args):
    with open('../vit/stats.pkl', 'rb') as f:
        stats = pickle.load(f)

    means = [stats['mean'][f'channel_{i}'] for i in [2,5,0,1,3,4,6]]
    stds = [stats['std'][f'channel_{i}'] for i in [2,5,0,1,3,4,6]]

    train_dataset = aia_euv(args.json_path, subset='training', transform=v2.Compose([
        CustomTransform(means, stds, zscore=True),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomVerticalFlip(p=0.5)
        ]))

    validation_dataset = aia_euv(args.json_path, subset='validation', transform=v2.Compose([
        CustomTransform(means, stds, zscore=True)
        ]))

    return train_dataset, validation_dataset

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
    parser.add_argument("-dummy-data", "--dummy-data",action='store_true', help="Train first on a small dataset sampled from the original")
    parser.add_argument("-num-samples", "--num-samples", default=150, help="""number of samples to take from original data;
            only valid if --dummy-data set to True""")

    args = parser.parse_args()
    args_dict = vars(args)
    print("Args dict:", args_dict)
    wandb.init(
        project= "flare_torch",
        config= args_dict
            )

    model = AlexNet(num_classes=1)

    train_ds, val_ds = prepare_dataset(args)

    print("train size", len(train_ds))
    print("validation size", len(val_ds))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}")

    torch.cuda.reset_peak_memory_stats()

    model.to(device)

    if args.dummy_data == True:
        print("Training on dummy data")
        train_loader = dummy_data(train_ds, num_samples=args.num_samples, 
                batch_size=args.batch_size)

        val_loader = dummy_data(val_ds, num_samples=args.num_samples, 
                batch_size=args.batch_size)

        n_steps_per_epoch = len(train_loader)

    else:
        train_loader = DataLoader(train_ds, batch_size = args.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size = args.batch_size, shuffle=False)

        n_steps_per_epoch = math.ceil(len(train_loader.dataset) / args.batch_size)

    print(f"Steps per epoch:{n_steps_per_epoch}")

    print(device)

    wandb_dir = wandb.run.name
    output_dir = os.path.join("output", wandb_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    start_time = time.time()
    train_loop(model, train_loader, val_loader, n_steps_per_epoch, output_dir, args, device=device)

    epoch_time = time.time() - start_time
    print(f"Time taken to run single epoch: {np.round(epoch_time,4)/60} mins")
