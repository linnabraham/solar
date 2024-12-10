#!/bin/env python
import tensorflow as tf
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, precision_score, recall_score
import numpy as np
from tf_utils import get_trained_model, get_parser, parse_json, parse_images
from helpers.alexnet import AlexNet
from train_alexnet import get_compiled_model

def evaluate_model_metrics(x_train, y_train):
    count = 0
    pred_scores = []
    ground_truth = []
    for image_paths, label in zip(x_train, y_train):
        if count > 50:
            break
        print("Label", label)
        ground_truth.append(label)
        images = parse_images(image_paths, args)
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

def print_weights(model):
    for item in model.trainable_variables:
        print(item.shape)
    first_layer_weights = model.trainable_variables[0]
    sliced_weights = first_layer_weights[:,:,0,0]
    print(sliced_weights)
    second_last_layer_weights = model.trainable_variables[-2:-1]
    last_layer_weights = model.trainable_variables[-1]
    print(second_last_layer_weights)
    print(last_layer_weights)

def plot_hist(model):
    first_layer_weights = model.trainable_variables[0]
    sliced_weights = first_layer_weights[:,:,0,0]
    plt.hist(sliced_weights.numpy().flatten(), bins=50)
    #plt.hist(sliced_weights.numpy().flatten(), range=(-0.05,0.05), bins=50)
    plt.show()


if __name__=="__main__":
    # force channels-first ordering
    from tensorflow.keras import backend
    import matplotlib.pyplot as plt
    backend.set_image_data_format('channels_first')
    gpu = tf.config.experimental.list_physical_devices('GPU')[0]
    tf.config.experimental.set_memory_growth(gpu, True)

    parser = get_parser()
    parser.add_argument("--json-path", help="json file with training data paths")
    parser.add_argument("--modelpath", help="path of trained model")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--stats-file", help="path of pickle file containing data mean and std")
    args = parser.parse_args()

    model = get_compiled_model(args)
    model.load_weights(args.modelpath)
    print_weights(model)
    plot_hist(model)
    plt.savefig("hist_trained_weights.png")
    plt.close()

    if not args.json_path is None:
        x_train, y_train, x_test, y_test = parse_json(args.json_path)
        x_train, y_train = shuffle(x_train, y_train, random_state=42)
        x_test, y_test = shuffle(x_test, y_test, random_state=42)
        print(len(x_train), len(x_test))

        evaluate_model_metrics(x_train, y_train)
