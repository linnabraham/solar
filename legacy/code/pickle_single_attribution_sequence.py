import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from aarp_ml.model import training
from aarp_ml.integrated_gradients import aarp_ig
from aarp_ml.dataset import aarp_dataset
import pickle
import argparse

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--trained_model_path', required=True)
    args = parser.parse_args()

    input_shape=(512,512)
    num_channels=7
    #trained_model_path = "outputs/electric-star-195/best_model.h5",
    stats_file=os.path.expanduser("~/july/solar/stats.pkl")
    dataset = aarp_dataset(json_path=os.path.expanduser("~/july/solar/solar_dataset.json"))
    test_ds = dataset.get_subset('test')

    alexnet = training(aarp_dataset, stats_file, input_shape, num_channels)
    model = alexnet.get_trained_model(
            trained_model_path = args.trained_model_path,
            ).model
    ig = aarp_ig(input_shape, num_channels, model)

    aarp_seq = test_ds.create_aarp_sequence(3291)
    attribution_seq = ig.get_attribution_sequence(aarp_seq)
    print("Pickling output to disk...")
    with open('attribution_seq_aarp_3291.pkl', 'wb') as file:
        pickle.dump(attribution_seq, file)
