import argparse
import tensorflow as tf
import os
import json
import numpy as np
from astropy.io import fits
from helpers.alexnet import AlexNet

def get_trained_model(args):
    """
    Return the model with learnt weights for inference tasks
    """
    height, width = args.input_shape
    classification_threshold = 0.5
    model = AlexNet.build(width=width, height=height, depth=7, classes=1, reg=0.0002)
    model.load_weights(args.modelpath)
    return model

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-input-shape', '--input-shape', nargs='+', type=int, default=(512,512))
    parser.add_argument('-num-channels', '--num-channels', default=7)
    return parser

def get_compiled_model(args):
    height, width = args.input_shape
    classification_threshold = 0.5

    METRICS = [
          tf.keras.metrics.Precision(thresholds=classification_threshold,
                                     name='precision'),
          tf.keras.metrics.Recall(thresholds=classification_threshold,
                                  name="recall"),
          tf.keras.metrics.AUC(num_thresholds=100, curve='PR', name='auc_pr'),
    ]
    model = AlexNet.build(width=width, height=height, depth=7, classes=1, reg=0.0002)

    print("[INFO] compiling model...")
    model.compile(loss="binary_crossentropy", optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), metrics=METRICS)
    return model

def parse_images(imgs:list, args):
    height, width = args.input_shape
    images = np.zeros((len(imgs), height, width))
    for i, img in enumerate(imgs):
        hdul = fits.open(img)
        image = hdul[0].data
        images[i,:,:] = image
    return images

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

def dataset_from_json(args):
    height, width = args.input_shape
    nchannels = args.num_channels
    x_train, y_train, x_val, y_val = parse_json(args.json_path)

    images = tf.data.Dataset.from_generator(generator = lambda: img_generator(x_train, args),
                                            output_types=tf.float32,
                                            output_shapes=[nchannels, height, width])
    labels = tf.data.Dataset.from_generator(generator = lambda: label_generator(y_train),
                                            output_types = tf.int32,
                                            output_shapes = ())

    train_ds = tf.data.Dataset.zip((images, labels))

    images = tf.data.Dataset.from_generator(generator = lambda: img_generator(x_val, args),
                                            output_types=tf.float32,
                                            output_shapes=[nchannels, height, width])
    labels = tf.data.Dataset.from_generator(generator = lambda: label_generator(y_val),
                                            output_types = tf.int32,
                                            output_shapes = ())
    val_ds = tf.data.Dataset.zip((images, labels))

    return train_ds, val_ds

