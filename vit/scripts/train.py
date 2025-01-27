import sys, os
sys.path.append(os.path.expanduser("~/july/solar/"))
from aarp_ml.torch.dataset import aia_euv, CustomTransform
from aarp_ml.torch.model import DeepFlare_ViT, SaveBestModel
import time
import argparse
import math
import wandb
from tqdm import tqdm
import pickle
from torchvision.transforms import v2
import torch
from torch.utils.data import DataLoader

def validate_model(model, val_dl, loss_func, device):
    model.eval()
    val_loss = 0.
    with torch.inference_mode():
        correct = 0
        TP = 0
        FP = 0
        TN = 0
        FN = 0
        for i, (images, labels) in tqdm(enumerate(val_dl), total=len(val_dl), leave=False):
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


def train_loop(train_dataset, train_loader, val_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    threshold = 5000  # GPU memory threshold measured in megabytes

    n_steps_per_epoch = math.ceil(len(train_loader.dataset) / args.batch_size)
    print(f"Length of training data", len(train_loader.dataset))
    print(f"Steps per epoch:{n_steps_per_epoch}")

    wandb_dir = wandb.run.name
    output_dir = os.path.join("output", wandb_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    save_best_model_callback = SaveBestModel(monitor='val_loss', mode='min')

    start_epoch = 0

    if args.resume:
        if args.modelpath:
            print(f"Loding saved model from f{args.modelpath}")
            checkpoint = torch.load(args.modelpath)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            #loss = checkpoint['loss']
        else:
            raise FileNotFoundError(f"Checkpoint file not found at {args.modelpath}")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        running_loss = 0.0

        start_time = time.time()

        print(f"Epoch:{epoch}")

        for step, (inputs, labels) in tqdm(enumerate(train_loader), total=len(train_loader), leave=False):

            current_memory = torch.cuda.memory_allocated() / (1024 ** 2)

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
        val_loss, accuracy, precision, recall = validate_model(model, val_loader, criterion, device)

        val_metrics = {"val/val_loss": val_loss, 
                       "val/val_accuracy": accuracy,
                       "val/precision":precision,
                       "val/recall":recall}

        wandb.log({**metrics, **val_metrics})

        save_best_model_callback(val_loss, model, os.path.join(output_dir,"trained_model.pth"))

        epoch_loss = running_loss / len(train_dataset)
        print(f"Epoch loss: {epoch_loss}")

        val_ds = aia_euv(args.json_path, subset='validation')

        ig_val_loader = DataLoader(val_ds, batch_size = 64, shuffle=True)
        #log_ig_attributes(model, ig_val_loader, batch_idx=0, channel=0)

        epoch_time = time.time() - start_time
        print(f"Time taken to run single epoch: {epoch_time/60} mins")

        max_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # Convert to megabytes
        print(f"Maximum GPU memory usage: {max_memory} MB")

        max_memory_reserved = torch.cuda.max_memory_reserved() / (1024 ** 2)
        print(f"Maximum GPU memory reserved: {max_memory_reserved}")

def train(args):
    args_dict = vars(args)
    project_name = "flare_torch"
    wandb.init(
        project= project_name,
        config= args_dict
            )

    vit_model = DeepFlare_ViT(height=512, n_classes=2, n_passbands=7)
    model = vit_model.model

    json_path = args.json_path
    stats_file = args.stats_file

    with open(stats_file, 'rb') as f:
        stats = pickle.load(f)

    means = [stats['mean'][f'channel_{i}'] for i in range(7)]
    stds = [stats['std'][f'channel_{i}'] for i in range(7)]

    train_dataset = aia_euv(json_path, subset='training', transform=v2.Compose([
        CustomTransform(means, stds),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomVerticalFlip(p=0.5)
        ]))

    validation_dataset = aia_euv(json_path, subset='validation', transform=v2.Compose([CustomTransform(means, stds)]))

    print("Checking data specifications")
    for i in range(len(train_dataset)):
        features, label = train_dataset[i]
        print(features.shape)
        break

    print("train size", len(train_dataset))
    print("validation size", len(validation_dataset))

    train_loader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle=True)

    val_loader = DataLoader(validation_dataset, batch_size = args.batch_size, shuffle=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}")

    torch.cuda.reset_peak_memory_stats()

    model.to(device)

    train_loop(train_dataset, train_loader, val_loader, model, device)

if __name__== "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-json-path", "--json-path")
    parser.add_argument("-stats-file", "--stats-file")
    parser.add_argument("-batch-size", "--batch-size", type=int, default=32)
    parser.add_argument("-epochs", "--epochs", type=int, default=5)
    parser.add_argument('-lr', '--lr', type=float, default=0.001)
    parser.add_argument('-resume', '--resume', action="store_true", help="Flag to resume training from a previous epoch")
    parser.add_argument('-modelpath', '--modelpath', help="location of saved model")

    args = parser.parse_args()
    train(args)
