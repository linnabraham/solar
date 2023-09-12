import json
import os
from astropy.io import fits
import tensorflow as tf
import numpy as np
from helpers.alexnet import AlexNet
from tensorflow.keras import backend

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
    json_path = "AARPS_Fixed_Size/dataset.json"
    with open(json_path) as f:
        data = json.load(f)

        x_train = [
                [os.path.join(os.path.dirname(json_path), c[str(i)]) for i in range(7)]
                 for c in data.get('training')
                 ]
        y_train = [p['label'] for p in data.get('training')]

    images = tf.data.Dataset.from_generator(generator = lambda: img_generator(x_train),
                                            output_types=tf.float32,
                                            output_shapes=[6, 512, 512])
    labels = tf.data.Dataset.from_generator(generator = lambda: label_generator(y_train),
                                            output_types = tf.int32,
                                            output_shapes = ())

    dataset = tf.data.Dataset.zip((images, labels))
    dataset = dataset.batch(1)


    # force channels-first ordering
    backend.set_image_data_format('channels_first')
    model = AlexNet.build(width=512, height=512, depth=6, classes=1, reg=0.0002)
    print("[INFO] compiling model...")
    model.compile(loss="binary_crossentropy", optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3))

    history = model.fit(dataset, epochs=10, shuffle=True)


