#!/bin/env python
import tensorflow as tf
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, precision_score, recall_score
import numpy as np
from tf_utils import get_trained_model, get_parser, dataset_from_json, parse_json, parse_images

def preprocess_images_E2(images):
    images = np.where(images<0, np.zeros_like(images), images)
    images = np.sqrt(images)
    return images

def preprocess_images_E6(images):
    images = np.where(images<0, np.zeros_like(images), images)+1
    images = np.log(images)
    return images

def preprocess_label_E2(label):
    return int(label)

def per_image_standardization(image):
    mean = np.mean(image)
    stddev = np.std(image)
    # Prevent division by zero
    return standardized_image

def evaluate_on_tf_datasets(args):
    train_ds, val_ds = dataset_from_json(args)
    train_ds = train_ds.shuffle(buffer_size=1000).batch(args.batch_size)
    train_ds = train_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    print("Iterating through dataset")
    for images_batch, labels_batch in train_ds:
        label = labels_batch[0].numpy()
        if label == 0:
            count += 1
        else:
            continue
        if count > 10:
            break
        prediction = model.predict(images_batch)
        print("Labels\n", label)
        print(prediction)
def evaluate_model_metrics(x_train, y_train):
    count = 0
    pred_scores = []
    ground_truth = []
    for image_paths, label in zip(x_train, y_train):
        if count > 50:
            break
        print("Label", label)
        label = preprocess_label_E2(label)
        ground_truth.append(label)
        images = parse_images(image_paths, args)
        images = preprocess_images_E2(images)
        images = per_image_standardization(images)
        images = np.expand_dims(images, axis=0)
        prediction = model.predict(images)
        pred_scores.append(prediction[0])
        print("Prediction", prediction)
        count += 1

    predicted = [ 1 if pred_score > 0.5 else 0 for pred_score in pred_scores ] 
    cm  = confusion_matrix(ground_truth, predicted)
    precision  = precision_score(ground_truth, predicted, zero_division=0.0)
    recall  = recall_score(ground_truth, predicted, zero_division=0.0)
    print(cm)
    print("Precision", precision)
    print("Recall", recall)

if __name__=="__main__":
    # force channels-first ordering
    from tensorflow.keras import backend
    backend.set_image_data_format('channels_first')
    gpu = tf.config.experimental.list_physical_devices('GPU')[0]
    tf.config.experimental.set_memory_growth(gpu, True)

    parser = get_parser()
    parser.add_argument("--json-path")
    parser.add_argument("--modelpath")
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()

    model = get_trained_model(args)
    x_train, y_train, x_test, y_test = parse_json(args.json_path)
    x_train, y_train = shuffle(x_train, y_train, random_state=42)
    x_test, y_test = shuffle(x_test, y_test, random_state=42)
    print(len(x_train), len(x_test))

    evaluate_model_metrics(x_train, y_train)


