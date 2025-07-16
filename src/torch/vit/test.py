import torch
from torch.utils.data import DataLoader
import argparse
import time
from aarp_ml.torch.dataset import aia_euv
from tqdm import tqdm
from torchvision.transforms import v2
from src.torch.vit.train import TrainingConfig
from src.torch.vit.utils import get_data_model
from sklearn.metrics import confusion_matrix

def main(*, model, val_dl, loss_func, device):
    """Validate model and compute metrics."""
    model.eval()
    val_loss = 0.
    y_true, y_pred = [], []

    with torch.inference_mode():
        for images, labels in tqdm(val_dl, leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            val_loss += loss_func(outputs, labels) * labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        print(f"Confusion Matrix:\n{cm}")
        TN, FP, FN, TP = cm.ravel()
        precision = TP/(TP+FP) if (TP+FP) > 0 else 0
        recall = TP/(TP+FN) if (TP+FN) > 0 else 0

        # Print confusion matrix metrics
        print(f"\nConfusion Matrix Stats:")
        print(f"TP: {TP}, FP: {FP}")
        print(f"FN: {FN}, TN: {TN}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}\n")
        print(f"Accuracy: {(TP + TN) / len(val_dl.dataset):.4f}")

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-batch-size", "--batch-size", type=int, default=32)
    parser.add_argument('-trained-model', '--trained-model')
    parser.add_argument("-subset", "--subset")
    parser.add_argument("--stats-file")
    parser.add_argument("--json-path")
    args = parser.parse_args()

    config = TrainingConfig(json_path="solar_dataset.json", stats_file="stats.pkl")
    config.trained_model_path = "outputs/glad-shape-197/trained_model.pth"
    metadata, model, transform, device = get_data_model(config)

    dataset = aia_euv(config.json_path, subset=args.subset, transform=v2.Compose([transform]))
    data_loader = DataLoader(dataset, batch_size = args.batch_size, shuffle=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}")
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    start = time.time()
    main(model=model, val_dl=data_loader, loss_func=criterion, device=device)
    print("Time taken in mins:", int((time.time() - start)/60))
