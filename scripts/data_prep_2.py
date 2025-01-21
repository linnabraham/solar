from aarp_ml.data_prep_new import extract_from_dir
if __name__=="__main__":

    pos_dir_7h = "/data/linn/E8/compressed/pos"
    neg_dir_7h = "/data/linn/E8/compressed/neg"
    pos_dir_single = "/data/linn/E8/extracted/pos"
    neg_dir_single = "/data/linn/E8/extracted/neg"
    extract_from_dir(pos_dir_7h, neg_dir_7h,
                     pos_dir_single, neg_dir_single, target_shape=(512,512))
