import pickle
import os, sys
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(base_path)
import aarp_ml

if __name__=="__main__":
    attbn_seq_dir = "/data/linn/intensities_with_attributions-electric-star-195"
    attbn_seq_files = os.listdir(attbn_seq_dir)
    for file_path in attbn_seq_files:
        if not file_path.endswith(".pkl"):
            print(f"Skipping {file_path}")
            continue
        with open(f"{attbn_seq_dir}/{file_path}", "rb") as file:
            attbn_seq = pickle.load(file)
            label = attbn_seq.aarp_sequence.label
            print(f"{label=}")
            # attbn_images = attbn_seq.attribution_sequence.get_images(passband=94)
            # print(attbn_images.shape)
            break