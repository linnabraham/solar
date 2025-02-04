import os
from ..config import np, tf
from tensorflow.keras import backend
from tensorflow.keras.callbacks import ModelCheckpoint, Callback
from astro_utils.general import read_fits_single
import wandb
from wandb.keras import WandbCallback
import json
import pickle
from tensorflow.keras.models import load_model
from sklearn.utils import shuffle
from .alexnet import AlexNet
from aarp_ml.dataset import aarp_dataset

INPUT_SHAPE = (512,512)
NUM_CHANNELS = 7

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

def rescale(image, label):
    image = tf.image.per_image_standardization(image)
    return image, label

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

def save_arguments(args, filename):
    with open(filename, 'w') as f:
        json.dump(vars(args), f)

def get_true_labels(tfds):
    labels = tfds.map(lambda x,y: y)
    labels = np.array(list(labels.as_numpy_iterator()))
    labels = labels.reshape(-1, 1)
    return labels

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

class ml_dataset:
    def __init__(self, json_path):
        self.json_path = json_path

    @property
    def json_data(self):
        with open(self.json_path) as f:
            data = json.load(f)
        return data

    def get_tfds(self, subset_name):
        aarp_ds = aarp_dataset(json_path=self.json_path)
        subset = aarp_ds.get_subset(subset_name)
        file_path_list = [ [ file_path_channel for file_path_channel in file_path.values()]
                             for file_path in subset.file_paths]
        labels_list = subset.labels
        file_path_list, labels_list = shuffle(file_path_list, labels_list, random_state=42)
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

class trained_model:
    def __init__(self, trained_model_path=None):
        self.trained_model_path = trained_model_path
        if self.trained_model_path is not None:
            if not os.path.exists(self.trained_model_path):
                raise ValueError("Trained model path doesn't exist")
        self.model_ = None

    @property
    def model(self):
        if self._model is None:
            self._model = load_model(self.trained_model_path)
        return self._model

    @model.setter
    def model(self, model_with_weights):
        self._model = model_with_weights

    @property
    def image_size(self):
        # Return pretty much every information about your model
        config = self.model.get_config()

        # Return a tuple of width, height and channels as the expected input shape
        batch_input_shape = config["layers"][0]["config"]["batch_input_shape"]
        return batch_input_shape[1:-1]

    def predict_single(self):
        raise NotImplementedError

    def predict_on_test(self, test_ds, threshold=0.5):
        true_labels = get_true_labels(test_ds)
        predictions = self.model.predict(test_ds)
        predicted_labels = np.array([ 1 if prediction > threshold else 0 for prediction in predictions ])
        return (predicted_labels, predictions)

class training:
    def __init__(self, json_path, stats_file, input_shape=INPUT_SHAPE, num_channels=NUM_CHANNELS, trained_model_path=None):
        self.input_shape = input_shape
        self.num_channels = num_channels
        self.stats_file = stats_file
        self.json_path = json_path
        self.aarp_dataset = aarp_dataset(json_path=self.json_path)
        self.trained_model_path = trained_model_path

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
        cosine_annealing_lr = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=1e-3,  # Start with a high LR
            decay_steps=10000,           # Total steps for one cycle
            alpha=0.0                    # Minimum learning rate as a fraction of initial LR (0.0 = 0)
        )
        print("[INFO] compiling model...")
        model.compile(loss="binary_crossentropy", optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), metrics=METRICS)
        return model

    def train(self, epochs, batch_size, output_prefix):
        gpu = tf.config.experimental.list_physical_devices('GPU')[0]
        tf.config.experimental.set_memory_growth(gpu, True)
        os.environ["WANDB_SILENT"] = "true"

        model = self.get_compiled_model()
        if self.trained_model_path is not None:
            if os.path.exists(self.trained_model_path):
                print("Loading weights from model file", self.trained_model_path)
                model.load_weights(self.trained_model_path)

        if not os.path.exists(output_prefix):
            raise FileNotFoundError("Output_prefix directory should already exist")

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
        train_ds = ml_dataset(self.json_path).get_tfds(subset_name="training")
        val_ds = ml_dataset(self.json_path).get_tfds(subset_name="validation")

        train_ds = (train_ds
                    .batch(batch_size)
                    .prefetch(buffer_size=AUTOTUNE)
                    )

        val_ds = val_ds.batch(batch_size)

        history = model.fit(train_ds, validation_data=val_ds,  verbose=1, epochs=epochs, shuffle=True, callbacks=[mc,hc,
            WandbCallback(save_model=(False),save_graph=(False))])

        wandb.finish()

    def get_trained_model(self, trained_model_path):
        model = self.get_compiled_model()
        model.load_weights(trained_model_path)
        tm = trained_model()
        tm.model = model
        return tm
