import os, sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
import argparse
from aarp_ml.model import training
from aarp_ml.dataset import aarp_dataset
"""
    ds  = aarp_dataset(json_path=os.path.expanduser("~/july/solar/solar_dataset.json"))
    train_sess = training(ds, stats_file="stats.pkl", input_shape=(512, 512), num_channels=7)
    train_sess.train(epochs=2, batch_size=64, output_prefix="new-outputs")
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-json-path', '--json-path', default="solar_dataset.json")
    parser.add_argument('-batch-size', '--batch-size', type=int, default=32)
    parser.add_argument('-epochs', '--epochs', type=int, default=150)
    parser.add_argument('--stats-file')
    parser.add_argument('--trained-model')
    parser.add_argument('--retrain', action="store_true")
    args = parser.parse_args()
    print(vars(args))

    ds  = aarp_dataset(json_path=args.json_path)
    if not args.retrain == True:
        train_sess = training(ds, stats_file=args.stats_file, input_shape=(512, 512), num_channels=7)
    else:
        train_sess = training(ds, stats_file=args.stats_file, input_shape=(512, 512), num_channels=7,
                              trained_model_path=args.trained_model)
    train_sess.train(epochs=args.epochs, batch_size=args.batch_size, output_prefix="outputs")
