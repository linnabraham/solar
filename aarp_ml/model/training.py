import numpy as np
import tensorflow as tf
from tensorflow.keras import backend
from .alexnet import AlexNet
from .tf_utils import read_stats

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


class training:
    def __init__(self, input_shape, num_channels):
        self.input_shape = input_shape
        self.num_channels = num_channels
        self.stats_file = None
        self.trained_model_path = None

    def get_compiled_model(self,  stats_file):
        height, width = self.input_shape
        self.stats_file = stats_file

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

        data_mean, data_std = read_stats(stats_file)
        model = add_custom_layers(model, data_mean = data_mean, data_std = data_std, input_shape=self.input_shape, num_channels=self.num_channels)
        print("[INFO] compiling model...")
        model.compile(loss="binary_crossentropy", optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), metrics=METRICS)
        return model

    def get_trained_model(self, trained_model_path, stats_file):
        self.trained_model_path = trained_model_path
        model = self.get_compiled_model(stats_file)
        model.load_weights(self.trained_model_path)
        return model
