from aarp_ml.dataset import aarp_dataset
from src.torch.vit.train import TrainingConfig

if __name__ == "__main__":
    config = TrainingConfig(json_path="solar_dataset.json", stats_file="stats.pkl")
    ds = aarp_dataset(config.json_path)
    for split in ['training', 'validation', 'test']:
        ds.get_subset(split).subset_info()
