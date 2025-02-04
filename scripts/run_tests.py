import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import unittest
import pandas as pd
import numpy as np
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
from aarp_ml.data_prep import data_prep
from aarp_ml.dataset import aarp_dataset
from aarp_ml.model import training
from aarp_ml.model.training import ml_dataset
from colorama import Fore, Style

def log_info(message):
    print(f"{Fore.CYAN}[INFO]{Style.RESET_ALL} {message}")

def log_warning(message):
    print(f"{Fore.YELLOW}[WARNING]{Style.RESET_ALL} {message}")

def log_success(message):
    print(f"{Fore.GREEN}[SUCCESS]{Style.RESET_ALL} {message}")

def log_error(message):
    print(f"{Fore.RED}[ERROR]{Style.RESET_ALL} {message}")

class TestDataPrep(unittest.TestCase):
    def setUp(self):
        goes_event_list = "./data/GOES_event_list.csv"
        harp_to_noaa = "./data/all_harps_with_noaa_ars.txt"
        aarp_full_urls = "./data/aarps_full_urlist.txt"
        self.json_path = "solar_dataset_xx.json"
        stats_file = "stats_E8.pkl"
        trained_model_path = "outputs/vivid-hill-231/best_model.h5"

        self.dp = data_prep(goes_event_list, harp_to_noaa, aarp_full_urls)
        self.goes_df = self.dp.goes_df
        self.ds  = aarp_dataset(json_path=self.json_path)
        self.train_sess = training(self.json_path, stats_file=stats_file, input_shape=(512,512), num_channels=7)
        self.trained_model = self.train_sess.get_trained_model(trained_model_path)

    def test_goes_df_columns_datetime(self):
        datetime_columns = ['event_date', 'start_time', 'peak_time', 'end_time']
        for column in datetime_columns:
            self.assertTrue(pd.api.types.is_datetime64_any_dtype(self.goes_df[column]),
                            f"Column '{column}' is not of datetime type")

    def test_show_ds(self):
        print(self.ds.show())
        train_ds = self.ds.get_subset(subset_name="training")
        val_ds = self.ds.get_subset(subset_name="validation")
        test_ds = self.ds.get_subset(subset_name="test")
        log_info(f"Training: No. of Flared ids:{len(train_ds.get_flared_ids())}")
        log_info(f"Training: No. of Non-Flared ids:{len(train_ds.unique_aarp_ids - train_ds.get_flared_ids())}")
        log_info(f"Validation: No. of Flared ids:{len(val_ds.get_flared_ids())}")
        log_info(f"Validation: No. of Non-Flared ids:{len(val_ds.unique_aarp_ids - val_ds.get_flared_ids())}")
        log_info(f"Testing: No. of Flared ids:{len(test_ds.get_flared_ids())}")
        log_info(f"Testing: No. of Non-Flared ids:{len(test_ds.unique_aarp_ids - test_ds.get_flared_ids())}")


    def test_data_shuffle(self):
        print(f"Testing if data is shuffled in the training and validation subsets")
        ml_ds = ml_dataset(self.json_path)

        train_ds = ml_ds.get_tfds(subset_name="training")
        train_ds = train_ds.shuffle(1000).take(32)
        train_ds = train_ds.batch(32)

        for image_batch, label_batch in train_ds:
            unique_values , counts = np.unique(label_batch.numpy(), return_counts=True)
            for unique_value, count in zip(unique_values, counts):
                print(f"{unique_value=}, {count=}")
                assert count > 0
            print(f"Ratio: 0:1, {counts[0]/counts[1]}")
            break

        val_ds = ml_ds.get_tfds(subset_name="validation")
        val_ds = val_ds.shuffle(1000).take(32)
        val_ds = val_ds.batch(32)
        for image_batch, label_batch in val_ds:
            unique_values , counts = np.unique(label_batch.numpy(), return_counts=True)
            for unique_value, count in zip(unique_values, counts):
                print(f"{unique_value=}, {count=}")
                assert count > 0
            print(f"Ratio: 0:1, {counts[0]/counts[1]}")
            break

    def test_model_predict(self):
        ml_ds = ml_dataset(self.json_path)
        train_ds = ml_ds.get_tfds(subset_name="training")
        train_ds = train_ds.shuffle(1000).take(32)
        train_ds = train_ds.batch(32)
        model = self.trained_model.model
        for image_batch, label_batch in train_ds:
            predictions = model.predict(image_batch)
            for true_label, pred in zip(label_batch.numpy(), predictions):
                log_info(f"Ground Truth: {true_label}, Prediction: {pred}")
            break

if __name__ == "__main__":
    unittest.main()
