import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
import unittest
import pandas as pd
import numpy as np
import json
from sklearn.utils import shuffle
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
import aarp_ml.data_prep
from aarp_ml.data_prep import data_prep
from aarp_ml.dataset import aarp_dataset
from aarp_ml.model import training
from aarp_ml.model.training import ml_dataset
from aarp_ml.model.training import img_generator, label_generator, get_balanced_lists
from colorama import Fore, Style
from aarp_ml.data_prep import remove_offlimb, process_table_on_disk, get_fov_limits, dir_to_json
from scripts.data_single import get_download_list
from aarp_ml.utils import parse_json, get_aarp_ids
from aarp_ml.dataset import get_filepaths_labels

aarp_ml.USE_CACHE = True

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
        self.goes_event_list = "./data/GOES_event_list.csv"
        self.harp_to_noaa = "./data/all_harps_with_noaa_ars.txt"
        self.aarp_full_urls = "./data/aarps_full_urlist.txt"
        self.data_dir_pos = "/data/linn/E8/extracted/pos"
        self.data_dir_neg = "/data/linn/E8/extracted/neg"
        self.json_path = "solar_dataset_xx.json"
        self.stats_file = "stats_E8.pkl"
        self.trained_model_path = "outputs/golden-meadow-240/best_model.h5"

    def test_goes_df_columns_datetime(self):
        datetime_columns = ['event_date', 'start_time', 'peak_time', 'end_time']
        for column in datetime_columns:
            self.assertTrue(pd.api.types.is_datetime64_any_dtype(self.goes_df[column]),
                            f"Column '{column}' is not of datetime type")


    def test_data_dir(self):
        metadata = dir_to_json(self.data_dir_pos, self.data_dir_neg)
        aarp_ids = get_aarp_ids(metadata)
        aarp_ids_train_pos = aarp_ids['train']['pos']
        aarp_ids_train_neg = aarp_ids['train']['neg']
        aarp_ids_val_pos = aarp_ids['val']['pos']
        aarp_ids_val_neg = aarp_ids['val']['neg']
        aarp_ids_test_pos = aarp_ids['test']['pos']
        aarp_ids_test_neg = aarp_ids['test']['neg']

        self.train_pos = set(aarp_ids_train_pos)
        self.train_neg = set(aarp_ids_train_neg)
        self.val_pos = set(aarp_ids_val_pos)
        self.val_neg = set(aarp_ids_val_neg)
        self.test_pos = set(aarp_ids_test_pos)
        self.test_neg = set(aarp_ids_test_neg)

        self.train_all = self.train_pos | self.train_neg
        self.val_all = self.val_pos | self.val_neg
        self.test_all = self.test_pos | self.test_neg

        """Ensure no overlap between positive and negative labels in each split."""
        self.assertFalse(self.train_pos & self.train_neg, "Overlap found within training split.")
        self.assertFalse(self.val_pos & self.val_neg, "Overlap found within validation split.")
        self.assertFalse(self.test_pos & self.test_neg, "Overlap found within test split.")

        """Ensure no overlap between train, validation, and test splits."""
        self.assertFalse(self.train_all & self.val_all, "Overlap found between training and validation splits.")
        self.assertFalse(self.train_all & self.test_all, "Overlap found between training and test splits.")
        self.assertFalse(self.val_all & self.test_all, "Overlap found between validation and test splits.")

        print("\n=== Dataset Split Sizes ===")
        print(f"Training Positives: {len(aarp_ids_train_pos)}, Training Negatives: {len(aarp_ids_train_neg)}")
        print(f"Validation Positives: {len(aarp_ids_val_pos)}, Validation Negatives: {len(aarp_ids_val_neg)}")
        print(f"Test Positives: {len(aarp_ids_test_pos)}, Test Negatives: {len(aarp_ids_test_neg)}")

        print("\n=== Dataset Unique AARPs ===")
        print(f"Training Positives: {len(self.train_pos)}, Training Negatives: {len(self.train_neg)}")
        print(f"Validation Positives: {len(self.val_pos)}, Validation Negatives: {len(self.val_neg)}")
        print(f"Test Positives: {len(self.test_pos)}, Test Negatives: {len(self.test_neg)}")

    def test_tfds(self):
        data = parse_json(self.json_path)
        file_paths, labels_list = get_filepaths_labels(data, subset_name="training")
        #file_paths, labels_list = shuffle(file_paths, labels_list, random_state=42)
        height, width = (512,512)
        nchannels = len(data.get('channels'))
        images = tf.data.Dataset.from_generator(generator = lambda: img_generator(file_paths),
                                                output_types=tf.float32,
                                                output_shapes=[nchannels, height, width])
        labels = tf.data.Dataset.from_generator(generator = lambda: label_generator(labels_list),
                                                output_types = tf.int32,
                                                output_shapes = ())
        train_ds = tf.data.Dataset.zip((images, labels))
        train_ds = train_ds.batch(32)
        print("Class distribution in original training set:")
        unique, counts = np.unique(labels_list, return_counts=True)
        class_ratios = dict(zip(unique, counts))
        for class_id, count in class_ratios.items():
            print(f"Class {class_id}: {count} samples")

        balanced_file_paths, balanced_labels = get_balanced_lists(file_paths, labels_list)
        balanced_file_paths, balanced_labels = shuffle(balanced_file_paths, balanced_labels, random_state=42)

        print("Class distribution in balanced training set:")
        unique, counts = np.unique(balanced_labels, return_counts=True)
        class_ratios = dict(zip(unique, counts))
        for class_id, count in class_ratios.items():
            print(f"Class {class_id}: {count} samples")

        sample = balanced_labels[:32]
        print(np.bincount(sample))

        images = tf.data.Dataset.from_generator(generator = lambda: img_generator(balanced_file_paths),
                                                output_types=tf.float32,
                                                output_shapes=[nchannels, height, width])
        labels = tf.data.Dataset.from_generator(generator = lambda: label_generator(balanced_labels),
                                                output_types = tf.int32,
                                                output_shapes = ())
        train_ds = tf.data.Dataset.zip((images, labels))
        train_ds = train_ds.batch(32)

        for image_batch, label_batch in train_ds:
            print(image_batch.shape, label_batch.shape)
            print(label_batch)
            label_arr = label_batch.numpy()
            print(np.bincount(label_arr))
            break

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
        self.train_sess = training(self.json_path, stats_file=self.stats_file, input_shape=(512,512), num_channels=7)
        self.trained_model = self.train_sess.get_trained_model(self.trained_model_path)
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
