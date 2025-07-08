import os, sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
from aarp_ml.dataset import aarp_dataset

def main():
    dataset = aarp_dataset("/home/linn/july/solar/solar_dataset.json")
    # train_ds = dataset.get_subset(subset_name='training')
    # aarp_seq = train_ds.create_aarp_sequence(aarp_id=1321)
    test_ds = dataset.get_subset(subset_name='test')
    aarp_id = 377
    passband = 171
    aarp_seq = test_ds.create_aarp_sequence(aarp_id=aarp_id)
    # aarp_seq_passband  = aarp_seq.get_images(passband=passband)
    # print(aarp_seq_passband.shape)
    aarp_seq.create_aarp_movie(passband=passband, sqrt=False)

if __name__=="__main__":
    main()
