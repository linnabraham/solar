import sys, os
sys.path.append(os.path.expanduser("~/july/solar/"))
import torch
from torch.utils.data import Dataset, DataLoader
import argparse
import time
# from torch_train import DeepFlare_ViT, aia_euv, CustomTransform
from aarp_ml.torch.dataset import aia_euv, CustomTransform
from aarp_ml.torch.model import DeepFlare_ViT
from tqdm import tqdm
import pickle
from torchvision.transforms import v2

def validate_model(model, val_dl, loss_func):
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
        print("No. of correct predictions")
        print(correct)
        print("Length of validation dataset:", len(val_dl.dataset))

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

        print(f"Precision:{precision}, Recall:{recall}")



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-batch-size", "--batch-size", type=int, default=32)
    parser.add_argument('-trained-model', '--trained-model')
    parser.add_argument("-subset", "--subset")
    parser.add_argument("--stats-file")
    parser.add_argument("--json-path")
    args = parser.parse_args()
    vit_model = DeepFlare_ViT(height=512, n_classes=2, n_passbands=7)
    model = vit_model.model

    subset = args.subset

    # read the mean and std computed over the whole data and pickled to disk
    with open(args.stats_file, 'rb') as f:
        stats = pickle.load(f)

    means = [stats['mean'][f'channel_{i}'] for i in range(7)]
    stds = [stats['std'][f'channel_{i}'] for i in range(7)]

    dataset = aia_euv(args.json_path, subset=subset, transform=v2.Compose([CustomTransform(means, stds)]))
    data_loader = DataLoader(dataset, batch_size = args.batch_size, shuffle=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}")
    model.load_state_dict(torch.load(args.trained_model))
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    start = time.time()
    validate_model(model, data_loader, criterion)
    print("Time taken in mins:", int((time.time() - start)/60))
