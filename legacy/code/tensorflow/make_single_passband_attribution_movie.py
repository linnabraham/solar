import os, sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
import pickle
from aarp_ml.integrated_gradients import attribution_sequence

if __name__=="__main__":
    # path_to_attribution = "/data/linn/attributions-northern-forest-193/attribution_seq_377.pkl"
    path_to_attribution = "/data/linn/attributions-northern-forest-193/attribution_seq_377.pkl"
    with open(path_to_attribution, "rb") as file:
        attribution_seq = pickle.load(file)
    aarp_id = attribution_seq.aarp_id
    passband = 171
    attribution_seq.make_attribution_movie(passband=passband, filename = f"first_attrib_movie_aarp_id_{aarp_id}_passband_{passband}.mp4")

