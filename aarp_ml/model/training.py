import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend
from tensorflow.keras.callbacks import ModelCheckpoint, Callback
from astro_utils.general import read_fits_single
import wandb
from wandb.keras import WandbCallback
import json
import pickle
from .alexnet import AlexNet

def parse_images(img_paths:list):
    images_list = []
    for img_path in img_paths:
        image = read_fits_single(img_path)
        images_list.append(image)
    return np.stack(images_list, axis=0)

def img_generator(collection):
    for element in collection:
        yield parse_images(element)

def label_generator(collection):
    for element in collection:
        yield element

def get_tfds(aarp_dataset, subset_name):
    subset = aarp_dataset.get_subset(subset_name)
    file_path_list = [ [ file_path_channel for file_path_channel in file_path.values()]
                         for file_path in subset.file_paths]
    labels_list = subset.labels
    subset._generate_sample_image()
    image_sample = subset.sample_image.get('data')
    height, width = image_sample.shape
    nchannels = len(subset.all_wavelengths)
    images = tf.data.Dataset.from_generator(generator = lambda: img_generator(file_path_list),
                                            output_types=tf.float32,
                                            output_shapes=[nchannels, height, width])
    labels = tf.data.Dataset.from_generator(generator = lambda: label_generator(labels_list),
                                            output_types = tf.int32,
                                            output_shapes = ())
    tfds = tf.data.Dataset.zip((images, labels))
    return tfds

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

def read_stats(pickle_path):
    with open(pickle_path, 'rb') as f:
        stats = pickle.load(f)

    means = [stats['mean'][f'channel_{i}'] for i in range(7)]
    stds = [stats['std'][f'channel_{i}'] for i in range(7)]
    return means, stds


def flip_augment(images, labels, seed):
    new_seed = tf.random.experimental.stateless_split((seed,seed), num=1)[0, :]

    images = tf.image.stateless_random_flip_left_right(images, seed=new_seed)
    images = tf.image.stateless_random_flip_up_down(images, seed=new_seed)

    return (images, labels)

class FlipAugment(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(FlipAugment, self).__init__(**kwargs)

    def call(self, images, seed=42, training=True):
        if training:
            seed = tf.random.experimental.stateless_split((seed, seed), num=1)[0, :]
            images = tf.image.stateless_random_flip_left_right(images, seed=seed)
            images = tf.image.stateless_random_flip_up_down(images, seed=seed)
        return images


def add_custom_layers(model, data_mean:list, data_std:list, input_shape, num_channels):
    """
    Add a normalization layer to standardize the data channel-wise
    """
    data_var = [np.square(item) for item in data_std]
    norm_layer = tf.keras.layers.Normalization(axis=1, mean=data_mean, variance=data_var)
    inputs = tf.keras.Input(shape=(num_channels,)+input_shape)
    flip_augment_layer = FlipAugment()
    flipped = flip_augment_layer(inputs, training=True)
    normed = norm_layer(flipped)
    log_transformed = LogTransformLayer()(normed)
    outputs = model(log_transformed)
    model = tf.keras.Model(inputs, outputs)
    return model

class LogTransformLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(LogTransformLayer, self).__init__(**kwargs)

    def call(self, inputs):
        return tf.math.sign(inputs) * tf.math.log(tf.math.abs(inputs) + 1)

class SaveHistoryCallback(Callback):
    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path
        self.history = {'loss': [], 'val_loss': [], 'auc_pr':[], 'val_auc_pr':[], 'val_precision':[], 'val_recall':[]}

    def on_epoch_end(self, epoch, logs=None):
        self.history['loss'].append(logs.get('loss'))
        self.history['val_loss'].append(logs.get('val_loss'))
        self.history['auc_pr'].append(logs.get('auc_pr'))
        self.history['val_auc_pr'].append(logs.get('val_auc_pr'))
        self.history['val_precision'].append(logs.get('val_precision'))
        self.history['val_recall'].append(logs.get('val_recall'))

        with open(self.file_path, 'w') as f:
            json.dump(self.history, f)

def save_arguments(args, filename):
    with open(filename, 'w') as f:
        json.dump(vars(args), f)

class training:
    def __init__(self, aarp_dataset, stats_file, trained_model_path, input_shape, num_channels):
        self.input_shape = input_shape
        self.num_channels = num_channels
        self.stats_file = stats_file
        self.trained_model_path = trained_model_path
        self.aarp_dataset = aarp_dataset

    def get_compiled_model(self):
        assert self.stats_file is not None
        height, width = self.input_shape

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

        data_mean, data_std = read_stats(self.stats_file)
        model = add_custom_layers(model, data_mean = data_mean, data_std = data_std, input_shape=self.input_shape, num_channels=self.num_channels)
        print("[INFO] compiling model...")
        model.compile(loss="binary_crossentropy", optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), metrics=METRICS)
        return model

    def train(self, epochs, batch_size, output_prefix):
        gpu = tf.config.experimental.list_physical_devices('GPU')[0]
        tf.config.experimental.set_memory_growth(gpu, True)
        os.environ["WANDB_SILENT"] = "true"

        model = self.get_compiled_model()

        if not os.path.exists(output_prefix):
            raise FileNotFoundError

        wandb.init(project="AARP_Train")
        outdir = os.path.join(output_prefix, wandb.run.name)
        if os.path.exists(outdir):
            raise ValueError(f"Output directory {outdir} already exists")

        model_path = os.path.join(outdir,"best_model.h5")
        history_path = os.path.join(outdir,'history.json')

        print("Monitoring val_loss for saving best model")
        mc = ModelCheckpoint(model_path, monitor='val_loss', \
                mode='min', verbose=1, save_best_only=True)
        hc = SaveHistoryCallback(history_path)

        AUTOTUNE = tf.data.AUTOTUNE
        train_ds = get_tfds(self.aarp_dataset, subset_name="training")
        val_ds = get_tfds(self.aarp_dataset, subset_name="validation")

        train_ds = (train_ds
                    .batch(batch_size)
                    .prefetch(buffer_size=AUTOTUNE)
                    )

        val_ds = val_ds.batch(batch_size)

        history = model.fit(train_ds, validation_data=val_ds,  verbose=1, epochs=epochs, shuffle=True, callbacks=[mc,hc,
            WandbCallback(save_model=(False),save_graph=(False))])

        wandb.finish()

    def evaluate(self, batch_size):
        gpu = tf.config.experimental.list_physical_devices('GPU')[0]
        tf.config.experimental.set_memory_growth(gpu, True)

        model = self.get_compiled_model()

        if not self.trained_model_path is None:
            model.load_weights(self.trained_model_path)
        else:
            raise ValueError("trained_model_path is not defined")

        val_ds = get_tfds(self.aarp_dataset, subset_name="validation")
        val_ds = val_ds.batch(batch_size)

        result = model.evaluate(val_ds)
