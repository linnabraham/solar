from aarp_dataset import aarp_dataset, aarp_sequence
from aarp_dataset import log_transform_flatten, plot_intensity_distribution
# from train_alexnet import get_compiled_model
from integrated_grads import get_attributions_mask
import tensorflow as tf
from tensorflow.keras import backend
from helpers.alexnet import AlexNet
from tf_utils import read_stats
import numpy as np
import pickle
import matplotlib.pyplot as plt

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

class attribution_sequence():
    def __init__(self, aarp_sequence):
        self.label = aarp_sequence.label
        self.aarp_id = aarp_sequence.aarp_id
        self.all_wavelengths = aarp_sequence.all_wavelengths
        self.data = []

    @property
    def images(self):
        return self.get_images()

    # TODO: this function is similar to that defined in aarp_dataset class
    def get_images(self, passband=None, non_negative=False, percentile_cutoff=None):
        images_ts = []
        for ts, image_dict in self.data:
            if passband is None:
                image_multiband = [image for image in image_dict.values()]
                images = np.array(image_multiband)
                # images = np.stack(list(image_dict.values()), axis=-1)  # Combine all passbands

            elif passband in image_dict:
                images = image_dict[passband]
            else:
                raise ValueError(f"Passband {passband} not found in data.")
            # TODO: decide the order of these operation
            if percentile_cutoff:
                threshold = np.percentile(images, percentile_cutoff)
                images = np.where(images < threshold, 0, images)
            if non_negative:
                images = np.where(images < 0, 0, images)

            images_ts.append(images)

        return np.array(images_ts)

class aarp_intensities_with_attribution:
    def __init__(self, aarp_sequence, attribution_sequence):
        self.label = aarp_sequence.label
        self.aarp_id = aarp_sequence.aarp_id
        self.all_wavelengths = aarp_sequence.all_wavelengths
        self.aarp_sequence = aarp_sequence
        self.attribution_sequence = attribution_sequence

class aarp_ig:
    def __init__(self, input_shape, num_channels, model=None):
        self.model = model
        self.input_shape = input_shape
        self.num_channels = num_channels

    def get_attribution_sequence(self, aarp_sequence):
        attribution_seq = attribution_sequence(aarp_sequence)
        attribution_data = []
        for timestamp, image_dict in aarp_sequence.data:
            image = np.stack(list(image_dict.values()), axis=0)
            # image = np.stack(list(image_dict.values()), axis=-1)

            attribution_mask = get_attributions_mask(image,
                                                    self.model,
                                                    aarp_sequence.label,
                                                    self.input_shape,
                                                    self.num_channels)
            attribution_mask_arr = attribution_mask.numpy()
            attribution_mask_dict = {}
            for idx in range(attribution_mask_arr.shape[-1]):

                attribution_mask_dict[aarp_sequence.all_wavelengths[idx]] = \
                        attribution_mask_arr[:,:,idx]
            attribution_data.append((timestamp, attribution_mask_dict))
            attribution_seq.data = attribution_data
        return attribution_seq

def plot_attribution_based_intensity_distribution(aarp_intensities_with_attribution, passband=None):
    intensity_seq  = aarp_intensities_with_attribution.aarp_sequence
    attribution_seq = aarp_intensities_with_attribution.attribution_sequence

    if passband:
        intensity_data = intensity_seq.get_images(passband=passband,
                                                  non_negative=True)
        attribution_data = attribution_seq.get_images(passband=passband)
    else:
        intensity_data = intensity_seq.get_images(non_negative=True)
        attribution_data = attribution_seq.images

    for percentile_level in (60, 80, 90, 99):
        attribution_threshold = np.percentile(attribution_data, percentile_level)
        thresholded_intensities = intensity_data [attribution_data > attribution_threshold]
        log_transformed = log_transform_flatten(thresholded_intensities)
        plt.figure()
        plot_intensity_distribution(log_transformed)

if __name__=="__main__":
    input_shape=(512,512)
    num_channels=7
    alexnet = training(input_shape, num_channels)
    model = alexnet.get_trained_model(
            trained_model_path = "outputs/electric-star-195/best_model.h5",
            stats_file="/data/linn/stats.pkl")

    dataset = aarp_dataset(json_path="solar_dataset.json")
    test_ds = dataset.get_subset('test')
    # aarp_seq = test_ds.create_aarp_sequence(3291)
    ig = aarp_ig(input_shape, num_channels, model)
    # attribution_seq = ig.get_attribution_sequence(aarp_seq)
    # print("Pickling output to disk...")
    # with open('attribution_seq_aarp_3291.pkl', 'wb') as file:
    #     pickle.dump(attribution_seq, file)
    for aarp_id in test_ds.unique_aarp_ids:
        print("Processing AARP ID:", aarp_id)
        aarp_seq = test_ds.create_aarp_sequence(aarp_id=aarp_id)
        attribution_seq = ig.get_attribution_sequence(aarp_seq)
        print("Pickling output to disk...")
        with open(f'/data/linn/attribution_seq_{aarp_id}.pkl', 'wb') as file:
            pickle.dump(attribution_seq, file)
