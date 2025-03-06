import pickle
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from aarp_ml.dataset import aarp_dataset
"""
Create AARP sequences from all AARPs in the test set and pickle the objects
"""

def main():
    dataset = aarp_dataset("/home/linn/july/solar/solar_dataset.json")
    test_ds = dataset.get_subset(subset_name='test')
    print(test_ds.unique_aarp_ids)
    for aarp_id in test_ds.unique_aarp_ids:
        print("Processing AARP ID:", aarp_id)
        aarp_seq = test_ds.create_aarp_sequence(aarp_id=aarp_id)
        print("Pickling output to disk...")
        with open(f'aarp_seq_{aarp_id}.pkl', 'wb') as file:
            pickle.dump(aarp_seq, file)

if __name__=="__main__":
    main()
