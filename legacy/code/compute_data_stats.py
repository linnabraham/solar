#!/bin/env python
"""
Script to compute the mean and standard deviation on the entire dataset
for normalizing data during model training
"""

import argparse
import tensorflow as tf
import pickle
from tf_utils import dataset_from_json, get_parser

def compute_mean_and_std(dataset):
    # Initialize variables to accumulate the sum and sum of squares
    sum_values = tf.zeros(shape=(7,), dtype=tf.float32)
    sum_squared_values = tf.zeros(shape=(7,), dtype=tf.float32)
    count = 0

    # Iterate over the dataset
    for batch in dataset:
        # Assuming batch[0] contains the features
        values = batch[0]
        # Move the channel dimension to last
        values = tf.transpose(values, perm=[0, 2, 3, 1])
        # Do not sum over the channel
        sum_values += tf.reduce_sum(values, axis=[0, 1, 2])
        sum_squared_values += tf.reduce_sum(tf.square(values), axis=[0, 1, 2])
        # Do not use the channel number for calculating count
        count += tf.reduce_prod(values.shape[:-1]).numpy()

    # Compute the mean
    mean = sum_values / count

    # Compute the variance
    variance = (sum_squared_values / count) - tf.square(mean)

    # Compute the standard deviation
    std = tf.sqrt(variance)

    return mean, std

if __name__=="__main__":
    parser = get_parser()
    parser.add_argument("--json-path")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    train_ds, __ = dataset_from_json(args)
    train_ds  = train_ds.batch(args.batch_size)

    data, label = next(iter(train_ds))
    print("Data shape:", data.shape)

    data_mean, data_std = compute_mean_and_std(train_ds)

    stats = {
            'mean' : {f'channel_{i}': data_mean.numpy()[i] for i in range(args.num_channels)},
            'std' : {f'channel_{i}': data_std.numpy()[i] for i in range(args.num_channels)}
            }
    print(stats)

    with open('stats.pkl', 'wb') as f:
        pickle.dump(stats, f)
