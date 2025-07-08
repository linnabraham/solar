import pandas as pd
import json
import os
from tqdm import tqdm
from aarp_ml.dataset import all_wavelengths
from src.torch.vit.train import TrainingConfig

def clean_json_file(json_path, output_path):
    """
    Removes entries from the dataset JSON where any file path does not exist.
    Preserves all original fields in each entry.
    Saves the cleaned JSON to output_path.
    """
    with open(json_path, 'r') as json_file:
        metadata = json.load(json_file)

    for split in ['training', 'validation', 'test']:
        print(f"Cleaning {split} split...")
        entries = metadata[split]
        print(f"{len(entries)=}")
        keep_entries = []
        for entry in tqdm(entries, desc=f"Cleaning {split} split"):
            missing = False
            for i in range(7):
                filepath = entry.get(str(i))
                if not filepath or not os.path.exists(filepath):
                    print(f"[{split}] No such file: {filepath}")
                    missing = True
                    break
            if not missing:
                keep_entries.append(entry)
        metadata[split] = keep_entries
        print(f"{len(keep_entries)}=")
        print(f"{split} split: {len(keep_entries)} entries kept")

    with open(output_path, 'w') as out_file:
        json.dump(metadata, out_file, indent=4)
    print(f"Cleaned JSON saved to {output_path}")

if __name__ == "__main__":
    config = TrainingConfig(json_path="solar_dataset.json", stats_file="stats.pkl")
    clean_json_file(config.json_path, "solar_dataset.cleaned.json")
