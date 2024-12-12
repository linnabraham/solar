import os, sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
import argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
from aarp_ml.model import *
from aarp_ml.dataset import aarp_dataset
from ml_utils import compute_bce_losses

def analyze_results_to_file(aarp_ds, trained_model, subset_name):
    file_paths_171 = aarp_ds.get_subset(subset_name).get_file_paths(passband=171)[:64]
    tfds = ml_dataset(aarp_ds).get_tfds(subset_name)
    tfds = tfds.take(64).batch(args.batch_size)

    threshold = 0.5
    predicted_labels, predictions = tm.predict_on_test(tfds, threshold=threshold)
    true_labels = get_true_labels(tfds)
    bce_losses = compute_bce_losses(true_labels, predicted_labels)
    results = np.column_stack((file_paths_171, predictions, bce_losses, true_labels, predicted_labels))
    results_dest = f"predictions_results_{threshold}.csv"
    if not os.path.exists(results_dest):
        np.savetxt(results_dest, results, delimiter=',', fmt='%s')

def analyze_on_test(tm, test_ds):
    predicted_labels, predictions = tm.predict_on_test(test_ds)
    true_labels = get_true_labels(test_ds)
    bce_loss = np.mean(compute_bce_losses(true_labels, predicted_labels))
    print(f"{bce_loss=}")
    cm = confusion_matrix(true_labels, predicted_labels)
    print(cm)

def evaluate_on_tfds(tm, test_ds):
    results = tm.model.evaluate(test_ds)
    for key, value in zip(tm.model.metrics_names, results):
        print(f"{key}: {value}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-json-path', '--json-path', default="solar_dataset.json")
    parser.add_argument('-batch-size', '--batch-size', type=int, default=32)
    parser.add_argument('--stats-file', required=True)
    parser.add_argument('--trained_model_path', required=True)

    args = parser.parse_args()

    aarp_ds  = aarp_dataset(json_path=args.json_path)

    test_ds = ml_dataset(aarp_ds).get_tfds("test")
    val_ds = ml_dataset(aarp_ds).get_tfds("validation")

    train_sess = training(aarp_ds, stats_file=args.stats_file, input_shape=(512, 512), num_channels=7)
    tm = train_sess.get_trained_model(args.trained_model_path)

    # test_ds = test_ds.take(64)
    # val_ds = val_ds.take(64)

    print("Evaluating on test data")
    evaluate_on_tfds(tm, test_ds.batch(args.batch_size))

    # print("Evaluating on test data with image standardization applied")
    # test_ds = test_ds.map(rescale)
    # evaluate_on_tfds(tm, test_ds.batch(args.batch_size))

    print("Evaluating on validation data")
    evaluate_on_tfds(tm, val_ds.batch(args.batch_size))

    print("Analyzing on test data")
    analyze_on_test(tm, test_ds)

    # print("Writing results of analyzis with filenames etc. to file")
    # analyze_results_to_file(aarp_ds, tm, 'test')
