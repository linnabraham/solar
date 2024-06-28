import json
import os
from astropy.io import fits
import numpy as np
import wandb
import sys
import logging
import argparse
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, Callback
from tensorflow.keras.layers import Normalization
from tensorflow.keras import backend
from wandb.keras import WandbCallback
from helpers.alexnet import AlexNet
from tf_utils import get_parser, read_stats, parse_images, SaveHistoryCallback
tf.get_logger().setLevel(logging.WARNING)

def img_generator(collection, args):
    for element in collection:
        yield parse_images(element, args)

def label_generator(collection):
    for element in collection:
        yield element

def parse_json(json_path):
    with open(json_path) as f:
        data = json.load(f)

        x_train = [
                [os.path.join(os.path.dirname(json_path), c[str(i)]) for i in range(7)]
                 for c in data.get('training')
                 ]
        y_train = [p['label'] for p in data.get('training')]

        x_val  = [
                [os.path.join(os.path.dirname(json_path), c[str(i)]) for i in range(7)]
                 for c in data.get('validation')
                 ]
        y_val = [p['label'] for p in data.get('validation')]
    return x_train, y_train, x_val, y_val

def dataset_from_json(json_path, args):
    height, width = args.input_shape

    x_train, y_train, x_val, y_val = parse_json(json_path)

    images = tf.data.Dataset.from_generator(generator = lambda: img_generator(x_train, args),
                                            output_types=tf.float32,
                                            output_shapes=[7, height, width])
    labels = tf.data.Dataset.from_generator(generator = lambda: label_generator(y_train),
                                            output_types = tf.int32,
                                            output_shapes = ())

    train_ds = tf.data.Dataset.zip((images, labels))

    images = tf.data.Dataset.from_generator(generator = lambda: img_generator(x_val, args),
                                            output_types=tf.float32,
                                            output_shapes=[7, height, width])
    labels = tf.data.Dataset.from_generator(generator = lambda: label_generator(y_val),
                                            output_types = tf.int32,
                                            output_shapes = ())

    val_ds = tf.data.Dataset.zip((images, labels))

    return train_ds, val_ds

def add_custom_layers(model, data_mean:list, data_std:list, args):
    """
    Add a normalization layer to standardize the data channel-wise
    """
    data_var = [np.square(item) for item in data_std]
    norm_layer = tf.keras.layers.Normalization(axis=1, mean=data_mean, variance=data_var)
    inputs = tf.keras.Input(shape=(args.num_channels,)+args.input_shape)
    normed = norm_layer(inputs)
    log_transformed = LogTransformLayer()(normed)
    outputs = model(log_transformed)
    model = tf.keras.Model(inputs, outputs)
    return model

class LogTransformLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(LogTransformLayer, self).__init__(**kwargs)

    def call(self, inputs):
        return tf.math.sign(inputs) * tf.math.log(tf.math.abs(inputs) + 1)

def get_compiled_model(args):

    height, width = args.input_shape

    # force channels-first ordering
    backend.set_image_data_format('channels_first')

    classification_threshold = 0.5

    METRICS = [
          tf.keras.metrics.Precision(thresholds=classification_threshold,
                                     name='precision'),
          tf.keras.metrics.Recall(thresholds=classification_threshold,
                                  name="recall"),
          tf.keras.metrics.AUC(num_thresholds=100, curve='PR', name='auc_pr'),
    ]
    model = AlexNet.build(width=width, height=height, depth=7, classes=1, reg=0.0002)
    data_mean, data_std = read_stats(args.stats_file)
    model = add_custom_layers(model, data_mean = data_mean, data_std = data_std, args=args)

    print("[INFO] compiling model...")
    model.compile(loss="binary_crossentropy", optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), metrics=METRICS)
    return model

def get_savepaths_wandb(output, create_dirs=False):
    wandb.init(project="AARP_Train")
    outdir = os.path.join(output, wandb.run.name)
    if create_dirs:
        if not os.path.exists(outdir):
            print("Creating directory for storing run info", outdir)
            os.mkdir(outdir)
    return outdir

def save_arguments(args, filename):
    with open(filename, 'w') as f:
        json.dump(vars(args), f)

def flip_augment(images, labels, seed):
    new_seed = tf.random.experimental.stateless_split((seed,seed), num=1)[0, :]

    images = tf.image.stateless_random_flip_left_right(images, seed=new_seed)
    images = tf.image.stateless_random_flip_up_down(images, seed=new_seed)

    return (images, labels)

def checkpoint_best_model(model_path):
    print("Monitoring val_loss for saving best model")
    mc = ModelCheckpoint(model_path, monitor='val_loss', \
            mode='min', verbose=1, save_best_only=True)
    return mc

def checkpoint_each_epoch():
    return ModelCheckpoint(filepath=os.path.join(outdir,"model_{epoch:02d}_{val_loss:.2f}.h5"))

def filter_class(dataset, class_label):
    return dataset.filter(lambda x, y: y == class_label)

def downsample_negatives(json_path, train_ds, val_ds):
    x_train, y_train, x_val, y_val = parse_json(json_path)

    nneg_train = y_train.count(0)
    npos_train = y_train.count(1)
    nneg_val = y_val.count(0)
    npos_val = y_val.count(1)

    train_ds_pos = filter_class(train_ds,1)
    train_ds_neg = filter_class(train_ds,0)
    train_ds_neg = train_ds_neg.take(npos_train)
    train_ds = train_ds_pos.concatenate(train_ds_neg)

    val_ds_pos = filter_class(val_ds,1)
    val_ds_neg = filter_class(val_ds, 0)
    val_ds_neg = val_ds_neg.take(npos_val)
    val_ds = val_ds_pos.concatenate(val_ds_neg)

    return train_ds, val_ds

if __name__=="__main__":
    gpu = tf.config.experimental.list_physical_devices('GPU')[0]
    tf.config.experimental.set_memory_growth(gpu, True)
    os.environ["WANDB_SILENT"] = "true"
    parser = get_parser()
    parser.add_argument('-json-path', '--json-path', default="solar_dataset.json")
    parser.add_argument('-batch-size', '--batch-size', type=int, default=32)
    parser.add_argument('-epochs', '--epochs', type=int, default=150)
    parser.add_argument('--stats-file')
    parser.add_argument('--modelpath')
    parser.add_argument('--eval', action='store_true')

    args = parser.parse_args()
    print(vars(args))

    json_path = args.json_path
    batch_size = args.batch_size
    epochs = args.epochs

    if not os.path.exists("outputs"):
        print("Creating directory for storing outputs across runs named outputs")
        os.mkdir("outputs")

    # get directory to store individual run info
    outdir = get_savepaths_wandb("outputs", create_dirs=True)

    model_path = os.path.join(outdir,"best_model.h5")
    history_path = os.path.join(outdir,'history.json')

    print("Using the following paths for saving best model and history:", model_path, history_path)

    print("Saving the command line arguments to cmdline_args.json")
    save_arguments(args, os.path.join(outdir,"cmdline_args.json"))

    train_ds, val_ds = dataset_from_json(json_path=json_path, args=args)
    x_train, y_train, x_val, y_val = parse_json(args.json_path)
    print(f"Length of train vs val:", len(x_train), len(x_val))
    print(f"Class imbalance in original data(train):", y_train.count(0)/y_train.count(1))
    train_ds, val_ds = downsample_negatives(args.json_path, train_ds, val_ds)

    model = get_compiled_model(args)

    mc = checkpoint_best_model(model_path=model_path)

    hc = SaveHistoryCallback(history_path)

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = (train_ds
                .map(lambda x, y: flip_augment(x, y, seed=42), num_parallel_calls=AUTOTUNE)
                .batch(batch_size)
                .prefetch(buffer_size=AUTOTUNE)
                )
    train_ds_one_batch = train_ds.take(1)

    val_ds = val_ds.batch(batch_size)
    val_ds_one_batch = val_ds.take(1)

    if args.eval:
        print("Evaluating pre-trained model")
        model.load_weights(args.modelpath)
        result = model.evaluate(val_ds_one_batch)
        print(dict(zip(model.metrics_names, result)))

    else:
        history = model.fit(train_ds_one_batch, validation_data=val_ds_one_batch,  verbose=1, epochs=epochs, shuffle=True, callbacks=[mc,hc,
            WandbCallback(save_model=(False),save_graph=(False))])
    wandb.finish()
