import json
from vit.scripts.train import TrainingConfig

def get_metadata(config:TrainingConfig):
    with open(config.json_path, 'r') as json_file:
        metadata = json.load(json_file)
    return metadata
