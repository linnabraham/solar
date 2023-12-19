
import os,sys
import json
from astropy.io import fits
import numpy as np
import tensorflow as tf
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

def rescale(image, label):
    image = tf.image.per_image_standardization(image)
    return image, label

def get_true_predicted_labels(dataset, model):
    test_inputs = dataset.map(lambda x,y: x)
    test_labels = dataset.map(lambda x,y: y)
    # a shape error occurs unless we use some batching
    test_inputs = test_inputs.batch(32)
    predictions = model.predict(test_inputs)
    threshold = classification_threshold
    predicted_labels = [ 1 if prediction > threshold else 0 for prediction in predictions ]
    true_labels = np.array(list(test_labels.as_numpy_iterator()))
    true_labels = true_labels.reshape(-1, 1)
    return true_labels, predicted_labels

def print_results(dataset, model):
    results = model.evaluate(dataset)
    print(results)

if __name__=="__main__":
    json_path = "solar_dataset.json"
    with open(json_path) as f:
        data = json.load(f)

        x_test = [
                [os.path.join(os.path.dirname(json_path), c[str(i)]) for i in range(7)]
                 for c in data.get('test')
                 ]
        y_test = [p['label'] for p in data.get('test')]

    images = tf.data.Dataset.from_generator(generator = lambda: img_generator(x_test),
                                            output_types=tf.float32,
                                            output_shapes=[7, 512, 512])
    labels = tf.data.Dataset.from_generator(generator = lambda: label_generator(y_test),
                                            output_types = tf.int32,
                                            output_shapes = ())

    dataset = tf.data.Dataset.zip((images, labels))


    # force channels-first ordering
    backend.set_image_data_format('channels_first')

    model = AlexNet.build(width=512, height=512, depth=7, classes=1, reg=0.0002)

    test_data = dataset.map(rescale)

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
    model.load_weights("outputs/best_model.h5")

    first_filenames = [test_item[0] for test_item in x_test]
    first_filenames = np.array(first_filenames)

    true_labels, predicted_labels = get_true_predicted_labels(dataset=test_data, model=model)
    predicted_labels = np.array(predicted_labels)
    results = np.column_stack((first_filenames, predicted_labels))

    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(true_labels, predicted_labels)
    print(cm)

    #test_data = test_data.batch(32)
    #print_results(test_data, model)

    #np.savetxt("predictions.csv", results, delimiter=',', fmt='%s' )
