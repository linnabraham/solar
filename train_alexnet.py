import json
import os
from astropy.io import fits
import tensorflow as tf
import numpy as np
from helpers.alexnet import AlexNet
from tensorflow.keras import backend
from tensorflow.keras.callbacks import ModelCheckpoint, Callback, TensorBoard
import wandb
from wandb.keras import WandbCallback
import sys

wandb.init(project="AARP_Train")

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

def read_fits(file_path):
    hdul = fits.open(file_path)
    return hdul[0].data

def _parse_images(imgs:list):
    #TODO: dont hardocode height and width
    images = np.zeros((len(imgs),512, 512))
    for i, img in enumerate(imgs):
        image = read_fits(file_path=img)
        images[i,:,:] = image
    return images

def img_generator(collection):
    for element in collection:
        yield _parse_images(element)

def label_generator(collection):
    for element in collection:
        yield element


if __name__=="__main__":
    json_path = "solar_dataset.json"
    with open(json_path) as f:
        data = json.load(f)

        x_train = [
                [os.path.join(os.path.dirname(json_path), c[str(i)]) for i in range(7)]
                 for c in data.get('training')
                 ]
        y_train = [p['label'] for p in data.get('training')]

    images = tf.data.Dataset.from_generator(generator = lambda: img_generator(x_train),
                                            output_types=tf.float32,
                                            output_shapes=[7, 512, 512])
    labels = tf.data.Dataset.from_generator(generator = lambda: label_generator(y_train),
                                            output_types = tf.int32,
                                            output_shapes = ())

    dataset = tf.data.Dataset.zip((images, labels))


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

    pid = os.getpid()
    output = "outputs"
    outdir = os.path.join(output,str(pid))
    os.makedirs(outdir)

    model_path = os.path.join(output,"best_model.h5")
    mc = ModelCheckpoint(model_path, monitor='val_loss', \
            mode='min', verbose=1, save_best_only=True)

    history_path = os.path.join(outdir,'history.json')
    hc = SaveHistoryCallback(history_path)


    # Define a function to accumulate the size of the dataset
    def accumulate_size(count, _):
            return count + 1

    # Initialize a counter
    initial_count = tf.constant(0)

    # Use the reduce method to count the elements in the dataset
    dataset_size = dataset.reduce(initial_count, accumulate_size)
    dataset_size = dataset_size.numpy()
    train_size = np.floor(0.8*dataset_size)
    print("Train spit length", train_size)
    train_data = dataset.take(train_size)
    val_data = dataset.skip(train_size)
    train_data = train_data.batch(8)
    val_data = val_data.batch(8)

    with tf.device('/CPU:0'):
        model = AlexNet.build(width=512, height=512, depth=7, classes=1, reg=0.0002)

        print("[INFO] compiling model...")
        model.compile(loss="binary_crossentropy", optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), metrics=METRICS)

        history = model.fit(train_data, validation_data=val_data,  epochs=50, shuffle=True, callbacks=[mc,hc, WandbCallback()])
