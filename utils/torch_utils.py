import argparse
import torch
from tqdm import tqdm

def global_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-input-shape', '--input-shape', nargs='+', type=int, default=(512,512))
    #parser.add_argument('-json-path', '--json-path', default="../solar_dataset.json")
    return parser

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
