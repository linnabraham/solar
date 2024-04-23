import json
import os
from astropy.io import fits
import numpy as np
import wandb
import sys
import logging
import argparse
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
tf.get_logger().setLevel(logging.WARNING)

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
    images = np.zeros((len(imgs), height, width))
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

def rescale(image, label):
    image = tf.image.per_image_standardization(image)
    return image, label

def sqrt_transform(image, label):
    image = tf.where(image < 0, tf.zeros_like(image), image)
    image = tf.math.sqrt(image)
    return image, label

def dataset_from_json(json_path):
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


    images = tf.data.Dataset.from_generator(generator = lambda: img_generator(x_train),
                                            output_types=tf.float32,
                                            output_shapes=[7, height, width])
    labels = tf.data.Dataset.from_generator(generator = lambda: label_generator(y_train),
                                            output_types = tf.int32,
                                            output_shapes = ())

    train_ds = tf.data.Dataset.zip((images, labels))

    images = tf.data.Dataset.from_generator(generator = lambda: img_generator(x_val),
                                            output_types=tf.float32,
                                            output_shapes=[7, height, width])
    labels = tf.data.Dataset.from_generator(generator = lambda: label_generator(y_val),
                                            output_types = tf.int32,
                                            output_shapes = ())

    val_ds = tf.data.Dataset.zip((images, labels))

    return train_ds, val_ds

def get_compiled_model(args):
    from tensorflow.keras import backend
    from helpers.alexnet import AlexNet

    height = args.input_shape[0]
    width = args.input_shape[1]

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

    print("[INFO] compiling model...")
    model.compile(loss="binary_crossentropy", optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), metrics=METRICS)
    return model

def get_savepaths(create_dirs=False):
    pid = os.getpid()
    output = "outputs"
    outdir = os.path.join(output,str(pid))
    if create_dirs:
        os.makedirs(outdir)

    model_path = os.path.join(output,"best_model.h5")

    history_path = os.path.join(outdir,'history.json')
    return model_path, history_path

if __name__=="__main__":
    import tensorflow_addons as tfa
    from wandb.keras import WandbCallback

    parser = argparse.ArgumentParser()
    parser.add_argument('-input_shape', nargs='+', type=int, default=(512,512))
    parser.add_argument('-json_path', default="solar_dataset.json")
    parser.add_argument('-batch_size', type=int, default=32)
    parser.add_argument('-epochs', type=int, default=150)

    args = parser.parse_args()

    json_path = args.json_path
    batch_size = args.batch_size
    epochs = args.epochs

    train_ds, val_ds = dataset_from_json(json_path=json_path, args=args)
    model = get_compiled_model(args)

    model_path, history_path = get_savepaths(create_dirs=True)

    mc = ModelCheckpoint(model_path, monitor='val_loss', \
            mode='min', verbose=1, save_best_only=True)

    hc = SaveHistoryCallback(history_path)

    #train_ds = train_ds.map(sqrt_transform).map(rescale).batch(128)
    #val_ds = val_ds.map(sqrt_transform).map(rescale).batch(128)
    train_ds = train_ds.map(rescale).batch(batch_size)
    val_ds = val_ds.map(rescale).batch(batch_size)

    history = model.fit(train_ds, validation_data=val_ds,  verbose=1, epochs=epochs, shuffle=True, callbacks=[mc,hc,
        WandbCallback(save_model=(False),save_graph=(False))])
    wandb.finish()
