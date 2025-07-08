import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
import pickle
import time
from aarp_ml.data_prep import create_json
from aarp_ml.dataset import aarp_dataset
from aarp_ml.model.training import ml_dataset, compute_mean_and_std

if __name__ == "__main__":
    st = time.time()
    pos_dir_single = "/data/linn/E8/extracted/pos"
    neg_dir_single = "/data/linn/E8/extracted/neg"

    json_filename = os.path.join(parent_dir, "solar_dataset.json")
    pickle_file = os.path.join(parent_dir, "stats.pkl")
    create_json(pos_dir_single, neg_dir_single, json_filename)

    ds  = aarp_dataset(json_path=json_filename)
    num_channels = 7
    with open(pickle_file, 'wb') as f:
        train_ds = ml_dataset(json_path=json_filename).get_tfds(subset_name="training")
        train_ds = train_ds.batch(256)

        data_mean, data_std = compute_mean_and_std(train_ds)
        stats = {
                'mean' : {f'channel_{i}': data_mean.numpy()[i] for i in range(num_channels)},
                'std' : {f'channel_{i}': data_std.numpy()[i] for i in range(num_channels)}
                }
        print(stats)

        pickle.dump(stats, f)

    print(f"Script ran for {time.time() - st} seconds")
