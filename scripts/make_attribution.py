import os, sys
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(base_path)
import argparse
import time
from aarp_ml.dataset import aarp_dataset
from aarp_ml.model import training
from aarp_ml.integrated_gradients.aarp_ig import get_attribution_sequence
from concurrent.futures import ThreadPoolExecutor
import tracemalloc
from functools import partial
import numpy as np

def process_aarp_id(aarp_id, dataset, model):
    """Function to process a single AARP ID."""
    print(f"Processing AARP ID: {aarp_id}")
    st = time.time()

    aarp_seq = dataset.create_aarp_sequence(aarp_id=aarp_id)
    attribution_seq = get_attribution_sequence(aarp_seq, model)

    print(f"AARP ID {aarp_id} processed in {time.time() - st} seconds")
    return aarp_id, attribution_seq
    # return aarp_id, aarp_seq


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument('--trained_model_path', required=False)
    args = parser.parse_args()
    args.trained_model_path = "outputs/curious-bush-242/best_model.h5"
    print(f"Using trained model:{args.trained_model_path}")
    
    stats_file=os.path.expanduser("~/july/solar/stats.pkl")
    args.json_path = os.path.expanduser("~/july/solar/solar_dataset.json")
    dataset = aarp_dataset(json_path=args.json_path)
    test_ds = dataset.get_subset('test')
    unique_id = args.trained_model_path.split("/")[1]
    dest_dir = f"/data/linn/attribution_output/{unique_id}"
    if not os.path.exists(dest_dir):
        print("Creating directory", dest_dir)
        os.makedirs(dest_dir, exist_ok=True)
    # else:
    #     raise ValueError(f"Directory {dest_dir} already exists")
    model = training(args.json_path, stats_file).get_trained_model(args.trained_model_path).model

    # num_threads = min(2, len(test_ds.unique_aarp_ids))
    # print(f"Starting multithreading with {num_threads} threads...")

    global_st = time.time()
    tracemalloc.start()

    # with ThreadPoolExecutor(max_workers=num_threads) as executor:
    #     results = list(executor.map(lambda aarp_id: process_aarp_id(aarp_id, test_ds, model), list(test_ds.unique_aarp_ids)[:2]))

    # aarp_seqs = [seq for aarp_id, seq in results]

    # with ThreadPoolExecutor(max_workers=num_threads) as executor:
    #     func = partial(get_attribution_sequence, model=model)
    #     results = list(executor.map(func, aarp_seqs[:2]))

    for aarp_id in test_ds.unique_aarp_ids:
        save_path = os.path.join(dest_dir,f"aarp_{aarp_id}.npz")
        if os.path.exists(save_path):
            print(f"Skipping AARP ID: {aarp_id} as it already exists.")
            continue
        print("Processing AARP ID:", aarp_id)
        st = time.time()
        aarp_seq = test_ds.create_aarp_sequence(aarp_id=aarp_id)
        attribution_seq = get_attribution_sequence(aarp_seq, model)
        aarp_images = aarp_seq.get_images()
        attbn_images = attribution_seq.get_images()
        print(f"{time.time()-st} seconds to create aarp sequence")
        np.savez(save_path, aarp_images=aarp_images, attbn_images=attbn_images, label=aarp_seq.label)
        #break

    print(f"All AARP IDs processed. Time taken: {time.time() - global_st} seconds")
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"Current memory usage: {current / 10**6} MB; Peak memory usage: {peak / 10**6} MB")
