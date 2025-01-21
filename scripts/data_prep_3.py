import os, sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
from aarp_ml.data_prep_new import dir_to_json

if __name__=="__main__":
    pos_single_dir = "/data/linn/E8/extracted/pos"
    neg_single_dir = "/data/linn/E8/extracted/neg"
    filename = "solar_dataset_xx.json"
    dir_to_json(pos_single_dir, neg_single_dir, filename)
