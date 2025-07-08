import pandas as pd
import json
import os
from tqdm import tqdm
from aarp_ml.dataset import all_wavelengths
from src.torch.vit.train import TrainingConfig

def check_all_files_in_json(json_path):
    """
    Checks all file paths in the dataset JSON for readability issues.
    Prints out any files that cannot be read.
    """
    with open(json_path, 'r') as json_file:
        metadata = json.load(json_file)

    for split in ['training', 'validation', 'test']:
        print(f"Checking {split} split...")
        df = pd.DataFrame(metadata[split])
        df = df.rename({str(i): all_wavelengths[i] for i in range(7)}, axis=1)
        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Checking {split} split"):
            for pb in all_wavelengths:
                filepath = row[pb]
                if not os.path.exists(filepath):
                    print(f"[{split}] No such file: {filepath}")
        print(f"{split} split: Completed")

if __name__ == "__main__":
    config = TrainingConfig(json_path="solar_dataset.json", stats_file="stats.pkl")
    check_all_files_in_json(config.json_path)
