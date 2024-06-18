import argparse
import tensorflow as tf
from helpers.alexnet import AlexNet

def get_trained_model(args):
    """
    Return the model with learnt weights for inference tasks
    """
    height, width = args.input_shape
    classification_threshold = 0.5
    model = AlexNet.build(width=width, height=height, depth=7, classes=1, reg=0.0002)
    model.load_weights(args.modelpath)
    return model

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-input-shape', '--input-shape', nargs='+', type=int, default=(512,512))
    return parser

def get_compiled_model(args):
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

