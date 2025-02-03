import os
import sys
import unittest
import pandas as pd
import numpy as np
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
from aarp_ml.data_prep import data_prep
from aarp_ml.dataset import aarp_dataset
from aarp_ml.model import training
from aarp_ml.model.training import ml_dataset

class TestDataPrep(unittest.TestCase):
    def setUp(self):
        goes_event_list = "./data/GOES_event_list.csv"
        harp_to_noaa = "./data/all_harps_with_noaa_ars.txt"
        aarp_full_urls = "./data/aarps_full_urlist.txt"
        self.json_path = "solar_dataset_xx.json"
        stats_file = "stats_E8.pkl"
        trained_model_path = "outputs/earthy-sun-165/best_model.h5"

        self.dp = data_prep(goes_event_list, harp_to_noaa, aarp_full_urls)
        self.goes_df = self.dp.goes_df
        self.ds  = aarp_dataset(json_path=self.json_path)
        self.train_sess = training(self.ds, stats_file=stats_file, input_shape=(512,512), num_channels=7)
        self.trained_model = self.train_sess.get_trained_model(trained_model_path)

    def test_goes_df_columns_datetime(self):
        datetime_columns = ['event_date', 'start_time', 'peak_time', 'end_time']
        for column in datetime_columns:
            self.assertTrue(pd.api.types.is_datetime64_any_dtype(self.goes_df[column]),
                            f"Column '{column}' is not of datetime type")

    def test_data_shuffle(self):
        print(f"Testing if data is shuffled in the training subset")
        ml_ds = ml_dataset(self.json_path)

        train_ds = ml_ds.get_tfds(subset_name="training")
        train_ds = train_ds.take(32)
        train_ds = train_ds.batch(32)

        for image_batch, label_batch in train_ds:
            print(image_batch.shape)
            print(label_batch.numpy())
            assert len(np.unique(label_batch.numpy())) != 1
            break

    def test_model_predict(self):
        ml_ds = ml_dataset(self.json_path)
        train_ds = ml_ds.get_tfds(subset_name="training")
        train_ds = train_ds.take(32)
        train_ds = train_ds.batch(32)
        model = self.trained_model.model
        model.predict(train_ds)

if __name__ == "__main__":
    unittest.main()
