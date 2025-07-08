#!/bin/env python
"""
This script does a prediction on the test data using the saved model
and creates a confusion matrix and plots the images of AARPS which are 
correctly predicted and also incorrectly predicted
"""

import os,sys
import json
from astropy.io import fits
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend
# force channels-first ordering
backend.set_image_data_format('channels_first')


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

def rescale(image, label):
    image = tf.image.per_image_standardization(image)
    return image, label

def parse_json(json_path):
    with open(json_path) as f:
        data = json.load(f)

        x_train = [
                [os.path.join(os.path.dirname(json_path), c[str(i)]) for i in range(7)]
                 for c in data.get('training')
                 ]
        y_train = [p['label'] for p in data.get('training')]
    return x_train, y_train

def create_dataset(x_train, y_train):

    images = tf.data.Dataset.from_generator(generator = lambda: img_generator(x_train),
                                            output_types=tf.float32,
                                            output_shapes=[7, 512, 512])
    labels = tf.data.Dataset.from_generator(generator = lambda: label_generator(y_train),
                                            output_types = tf.int32,
                                            output_shapes = ())

    dataset = tf.data.Dataset.zip((images, labels))
    return dataset
   
def get_sizes(x_train, train_frac, val_frac):
    train_size = np.floor(0.6 * len(x_train))
    val_size = np.floor(0.2 * len(x_train))
    test_size = len(x_train)-(train_size+val_size)
    return train_size, val_size, test_size

def get_train_val_test(dataset, train_size, val_size):

    train_data = dataset.take(train_size)
    rest_data = dataset.skip(train_size)
    val_data = rest_data.take(val_size)
    test_data = rest_data.skip(val_size)

    return train_data, val_data, test_data

def get_trained_model(weights_path):
    # add the base path to the system path
    sys.path.append(base_path)
    from helpers.alexnet import AlexNet
    model = AlexNet.build(width=512, height=512, depth=7, classes=1, reg=0.0002)
    classification_threshold = 0.5

    METRICS = [
          tf.keras.metrics.Precision(thresholds=classification_threshold,
                                     name='precision'),
          tf.keras.metrics.Recall(thresholds=classification_threshold,
                                  name="recall"),
          tf.keras.metrics.AUC(num_thresholds=100, curve='PR', name='auc_pr'),
    ]

    print("[INFO] compiling model...")
    model.compile(loss="binary_crossentropy", optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), metrics=METRICS)
    model.load_weights(weights_path)
    return model

def get_true_labels(dataset):
    test_labels = dataset.map(lambda x,y: y)
    true_labels = np.array(list(test_labels.as_numpy_iterator()))
    true_labels = true_labels.reshape(-1, 1)
    return true_labels

def get_predicted_labels(predictions, threshold):
    predicted_labels = [ 1 if prediction > threshold else 0 for prediction in predictions ]
    return np.array(predicted_labels)

def print_confusion(true_labels, predicted_labels):
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(true_labels, predicted_labels)
    print(cm)

def get_test_filenames(x_train, train_size, val_size):
    test_items = x_train[int(train_size+val_size):]
    first_file_names = [test_item[0] for test_item in test_items]
    return np.array(first_file_names)


def create_plot_pdf(file_paths):
    import matplotlib.pyplot as plt
    import matplotlib
    import sunpy.visualization.colormaps as cm
    from matplotlib.backends.backend_pdf import PdfPages
    sdoaia94 = matplotlib.colormaps['sdoaia94']
    with PdfPages('output_plots.pdf') as pdf:
        count = 0
        for file_path in file_paths:
            data = read_fits(file_path)
            fig1, ax1 = plt.subplots()
            ax1.imshow(data, cmap=sdoaia94)
            ax1.set_title(f'Plot {count} - {file_path}',fontsize=9)
            count +=1
            pdf.savefig(fig1)



if __name__=="__main__":
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    # define the base path to be 1 level up from the location of file
    base_path = os.path.abspath(os.path.join(cur_dir,".."))
    json_path = os.path.join(base_path,"solar_dataset.json")
    weights_path = os.path.join(base_path,"outputs/best_model.h5")
    x_train, y_train  = parse_json(json_path)
    dataset = create_dataset(x_train, y_train)
    train_size, val_size, test_size = get_sizes(x_train, train_frac=0.6, val_frac=0.2)
    train_data, val_data, test_data = get_train_val_test(dataset, train_size, val_size)
    test_inputs = test_data.map(lambda x,y: x)
    # a shape error occurs unless we use some batching
    test_inputs = test_inputs.batch(32)
    true_labels = get_true_labels(test_data)
    model = get_trained_model(weights_path=weights_path)
    predictions = model.predict(test_inputs)
    predicted_labels = get_predicted_labels(predictions, threshold=0.5)
    print_confusion(true_labels, predicted_labels)
    test_filenames = get_test_filenames(x_train, train_size, val_size)
    failed = test_filenames[np.where(predicted_labels.flatten() != true_labels.flatten())]
    correct = test_filenames[np.where(predicted_labels.flatten() == true_labels.flatten())]
    create_plot_pdf(failed)
    #create_plot_pdf(correct[:50])
