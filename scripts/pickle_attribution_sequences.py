import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pickle
import argparse
from aarp_ml.model import training
from aarp_ml.integrated_gradients.aarp_ig import get_attribution_sequence
from aarp_ml.dataset import aarp_dataset
from aarp_ml.integrated_gradients.aarp_ig import aarp_intensities_with_attribution
"""
Create attribution sequence for all AARPs in a given subset
using a given model and pickle objects to disk inside a directory
named with the model code name
"""

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--trained_model_path', required=False)
    args = parser.parse_args()

    input_shape=(512,512)
    num_channels=7
    if args.trained_model_path is None:
        args.trained_model_path = "outputs/electric-star-195/best_model.h5"
    print(f"Using trained model:{args.trained_model_path}")

    stats_file=os.path.expanduser("~/july/solar/stats.pkl")
    dataset = aarp_dataset(json_path=os.path.expanduser("~/july/solar/solar_dataset.json"))
    test_ds = dataset.get_subset('test')


    unique_id = args.trained_model_path.split("/")[1]
    dest_dir = f"/data/linn/intensities_with_attributions-{unique_id}"
    if not os.path.exists(dest_dir):
        print("Creating directory", dest_dir)
        os.mkdir(dest_dir)
    else:
        raise ValueError(f"Directory {dest_dir} already exists")

    model = training(dataset, stats_file).get_trained_model(args.trained_model_path).model
    for aarp_id in test_ds.unique_aarp_ids:
        print("Processing AARP ID:", aarp_id)
        aarp_seq = test_ds.create_aarp_sequence(aarp_id=aarp_id)
        attribution_seq = get_attribution_sequence(aarp_seq, model)
        aarp_int_with_attbn = aarp_intensities_with_attribution(
                aarp_seq, attribution_seq)
        print("Pickling output to disk...")
        with open(f'{dest_dir}/intensities_with_attribution_seq_{aarp_id}.pkl', 'wb') as file:
            pickle.dump(aarp_int_with_attbn, file)
