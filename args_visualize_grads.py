import os,sys
import json
from astropy.io import fits
import numpy as np
import tensorflow as tf
from helpers.alexnet import AlexNet
from tensorflow.keras import backend
from math import log

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import sunpy.visualization.colormaps as cm
sdoaia131 = matplotlib.colormaps['sdoaia131']#
import json
import os,sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from astropy.io import fits

import tensorflow as tf
import logging
#tf.get_logger().setLevel(logging.ERROR)
import sys
sys.path.append('..')
from tensorflow.keras import backend
from helpers.alexnet import AlexNet
#from functools import lru_cache
import argparse

def cross_entropy(p, q):
     eps = 1e-15
     return -sum([p[i]*log(q[i]+eps) for i in range(len(p))])

def rescale(image, label):
    image = tf.image.per_image_standardization(image)
    return image, label

def get_trained_model(METRICS):
    model = AlexNet.build(width=512, height=512, depth=7, classes=1, reg=0.0002)
    print("[INFO] compiling model...")
    model.compile(loss="binary_crossentropy", optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), metrics=METRICS)
    model.load_weights("outputs/best_model.h5")
    return model

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

def interpolate_images(baseline,
                       image,
                       alphas):
  alphas_x = alphas[:, tf.newaxis, tf.newaxis, tf.newaxis]
  baseline_x = tf.expand_dims(baseline, axis=0)
  #baseline_x = tf.cast(baseline_x, dtype=tf.float64)

  input_x = tf.expand_dims(image, axis=0)
  input_x = tf.cast(input_x, dtype=tf.float32)
  delta = input_x - baseline_x
  images = baseline_x +  alphas_x * delta
  return images

def compute_gradients(model, images, target_class_idx):
  with tf.GradientTape() as tape:
    tape.watch(images)
    probs = model(images)[target_class_idx]
    #probs = tf.nn.softmax(logits, axis=-1)[:, target_class_idx]
  return tape.gradient(probs, images)

def integral_approximation(gradients):
  # riemann_trapezoidal
  grads = (gradients[:-1] + gradients[1:]) / tf.constant(2.0)
  integrated_gradients = tf.math.reduce_mean(grads, axis=0)
  return integrated_gradients

@tf.function
def one_batch(model, baseline, image, alpha_batch, target_class_idx):
    # Generate interpolated inputs between baseline and input.
    interpolated_path_input_batch = interpolate_images(baseline=baseline,
                                                       image=image,
                                                       alphas=alpha_batch)

    # Compute gradients between model outputs and interpolated inputs.
    gradient_batch = compute_gradients(model=model,
                                       images=interpolated_path_input_batch,
                                       target_class_idx=target_class_idx)
    return gradient_batch

def integrated_gradients(model,
                         baseline,
                         image,
                         target_class_idx,
                         m_steps=50,
                         batch_size=32):
  # Generate alphas.
  alphas = tf.linspace(start=0.0, stop=1.0, num=m_steps+1)

  # Collect gradients.
  gradient_batches = []

  # Iterate alphas range and batch computation for speed, memory efficiency, and scaling to larger m_steps.
  for alpha in tf.range(0, len(alphas), batch_size):
    from_ = alpha
    to = tf.minimum(from_ + batch_size, len(alphas))
    alpha_batch = alphas[from_:to]
    gradient_batch = one_batch(model, baseline, image, alpha_batch, target_class_idx)
    gradient_batches.append(gradient_batch)

  # Concatenate path gradients together row-wise into single tensor.
  total_gradients = tf.concat(gradient_batches, axis=0)

  # Integral approximation through averaging gradients.
  avg_gradients = integral_approximation(gradients=total_gradients)

  # Scale integrated gradients with respect to input.
  integrated_gradients = (image - baseline) * avg_gradients

  return integrated_gradients

def get_attributions_mask(images, model):

    #images_org = images.copy()
    #images = tf.image.per_image_standardization(images)

    m_steps=50
    alphas = tf.linspace(start=0.0, stop=1.0, num=m_steps+1) # Generate m_steps intervals for integral_approximation() below.

    baseline = tf.zeros(shape=(7, 512, 512))
    interpolated_images = interpolate_images(baseline, images, alphas=alphas)

    path_gradients = compute_gradients(
        model = model,
        images=interpolated_images,
        target_class_idx=1)

    pred = model(interpolated_images)

    ig = integral_approximation(
        gradients=path_gradients)

    ig_attributions = integrated_gradients(model=model,
                                           baseline=baseline,
                                           image=images,
                                           target_class_idx=1,
                                           m_steps=240)
    #image_channel = images_org[channel]
    attributions = np.moveaxis(ig_attributions, 0, 2)
    attribution_mask = tf.reduce_sum(tf.math.abs(attributions), axis=-1)
    #attribution_mask = tf.math.abs(attributions)
    return attribution_mask


if __name__=="__main__":
    json_path = "solar_dataset.json"
    with open(json_path) as f:
        data = json.load(f)

        x_test = [
                [os.path.join(os.path.dirname(json_path), c[str(i)]) for i in range(7)]
                 for c in data.get('test')
                 ]
        y_test = [p['label'] for p in data.get('test')]
        aarpid_test = [p['aarp_id'] for p in data.get('test')]
        ts_test = [p['timestamp'] for p in data.get('test')]

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

    model = get_trained_model(METRICS)

    # p is true labels
    p = [ int(label) for label in y_test]
    #q = predictions
    count = 0
    channel = 1
    alpha = 0.6
    for row, label, aarpid, ts  in zip(x_test, y_test, aarpid_test, ts_test):
        if label == "0":
            print("Skipping because label is 0")
            continue
        print(type(label))
        #images = tf.data.Dataset.from_generator(generator = lambda: img_generator(row),
        #        output_types = tf.float32,
        #        output_shapes = [7, 512, 512])

        images = _parse_images(row)
        images_pre = images.copy()
        print(images.shape)
        images = tf.image.per_image_standardization(images)
        attribution_mask  = get_attributions_mask(images, model) # channel=channel vmax=1000, cmap=sdoaia131, alpha=0.4
        print("Shape of attribution mask", attribution_mask.shape)
        sys.exit(0)
        mask_max = np.max(attribution_mask)
        mask_min = np.min(attribution_mask)

        if channel==1:
            cmap=sdoaia131

        fig, axes = plt.subplots(1, 3, figsize=(30, 30))

        plt.subplot(1,  3, 1)

        plt.imshow(attribution_mask, vmax = 0.2*mask_max, cmap=plt.cm.jet)
        plt.imshow(images_pre[1], vmax=1000, cmap=cmap, alpha=alpha)

        plt.subplot(1,  3, 2)
        plt.imshow(attribution_mask, vmax = 0.2*mask_max, cmap=plt.cm.jet)

        plt.subplot(1,  3, 3)
        plt.imshow(np.sqrt(images_pre[1]), cmap=cmap)
        plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
        fig.tight_layout()

        images = np.expand_dims(images, axis=0)
        prediction = model.predict(images)
        #print(prediction)
        expected = [ 1.0 - int(label), int(label)]
        predicted = [ 1.0 - prediction, prediction]
        ce = cross_entropy(expected, predicted)
        ce = np.abs(ce)

        plt.savefig(f"overlay_mask_{ce:.5f}_{count}_{aarpid}_{ts}_{alpha}_.png", bbox_inches="tight")
        #dataset = tf.data.Dataset.from_tensor_slices((images, label))
        count += 1

