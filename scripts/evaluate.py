import os, sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
import argparse
from aarp_ml.model import training
from aarp_ml.dataset import aarp_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-json-path', '--json-path', default="solar_dataset.json")
    parser.add_argument('-batch-size', '--batch-size', type=int, default=32)
    parser.add_argument('--stats-file', required=True)
    parser.add_argument('--trained_model_path', required=True)

    args = parser.parse_args()
    print(vars(args))

    ds  = aarp_dataset(json_path=args.json_path)
    train_sess = training(ds, stats_file=args.stats_file, trained_model_path = args.trained_model_path, input_shape=(512, 512), num_channels=7)
    train_sess.evaluate(args.batch_size)
