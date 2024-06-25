#!/bin/env python

"""
Script that reads the json metadata file used for training
Iterates through the list of testing images
Predicts on each of the 7 passband images and computes the cross-entropy loss
Plots the Integrated Gradients attribution mask for each channel
Saves it as a pdf file to directory based on the true label of the sample
"""

import tensorflow as tf
from helpers.alexnet import AlexNet
from tensorflow.keras import backend
from math import log
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import matplotlib
import sunpy.visualization.colormaps as cm
import json
import os,sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from astropy.io import fits
#from functools import lru_cache
import argparse
import tempfile
import random
from tf_utils import get_parser

def cross_entropy(label, prediction):
    # compute the cross-entropy loss for the sample
    p = [ 1.0 - int(label), int(label)]
    q = [ 1.0 - prediction, prediction]
    eps = 1e-15
    return -sum([p[i]*log(q[i]+eps) for i in range(len(p))])

def rescale(image, label):
    image = tf.image.per_image_standardization(image)
    return image, label

def get_trained_model(modelpath):
    global height
    global width
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
    model.load_weights(modelpath)
    return model

def read_fits(file_path):
    hdul = fits.open(file_path)
    return hdul[0].data

def _parse_images(imgs:list):
    global height
    global width
    images = np.zeros((len(imgs),height, width))
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

def get_attributions_mask(images, model, args):

    #images_org = images.copy()
    #images = tf.image.per_image_standardization(images)

    m_steps=50
    alphas = tf.linspace(start=0.0, stop=1.0, num=m_steps+1) # Generate m_steps intervals for integral_approximation() below.
    height, width = args.input_shape
    nchannels = args.num_channels
    baseline = tf.zeros(shape=(nchannels, height, width))
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
    #attribution_mask = tf.reduce_sum(tf.math.abs(attributions), axis=-1)
    attribution_mask = tf.math.abs(attributions)
    return attribution_mask

def plot_single_channel_attribution(attribution_mask, images_pre, channel=1):
    mask_max = np.max(attribution_mask)
    mask_min = np.min(attribution_mask)

    aia_cmaps = {0:'sdoaia94',
                    1:'sdoaia131',
                    2:'sdoaia171',
                    3:'sdoaia193',
                    4:'sdoaia211',
                    5:'sdoaia304',
                    6:'sdoaia335'}

    cmap = matplotlib.colormaps[aia_cmaps[channel]]

    fig, axes = plt.subplots(1, 3, figsize=(15, 15))

    plt.subplot(1,  3, 1)

    plt.imshow(attribution_mask, origin='lower', vmax = 0.2*mask_max, cmap=plt.cm.jet)
    plt.colorbar(shrink=0.5)
    plt.imshow(images_pre[channel], origin='lower', vmax=1000, cmap=cmap, alpha=alpha)
    plt.title("Overlayed")

    plt.subplot(1,  3, 2)
    plt.imshow(attribution_mask, origin='lower', vmax = 0.2*mask_max, cmap=plt.cm.jet)
    plt.colorbar(shrink=0.5)
    plt.title("Attribution mask")

    plt.subplot(1,  3, 3)
    plt.imshow(images_pre[channel], origin='lower', cmap=cmap)
    plt.colorbar(shrink=0.5)
    plt.title(f"Original: {aia_cmaps[channel]}")
    plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
    fig.tight_layout()
    return fig

def preprocess_data_E4(images):
    # apply the sqrt transform that is done during training
    images = np.where(images<0, np.zeros_like(images), images)
    images = np.sqrt(images)

    # create a copy of the images before standardizing for visual plotting
    images_pre = images.copy()

    images = tf.image.per_image_standardization(images)
    images = np.expand_dims(images, axis=0)
    return images, images_pre

def preprocess_data(images):
    # apply the sqrt transform that is done during training
    images = np.where(images<0, np.zeros_like(images), images)+1
    images = np.log(images)

    # create a copy of the images before standardizing for visual plotting
    images_pre = images.copy()

    images = tf.image.per_image_standardization(images)
    images = np.expand_dims(images, axis=0)
    return images, images_pre

if __name__=="__main__":
    parser = get_parser()
    parser.add_argument('-json-path', '--json-path')
    parser.add_argument('-modelpath', '--modelpath')
    args = parser.parse_args()

    height, width = args.input_shape

    # force channels-first ordering
    backend.set_image_data_format('channels_first')


    with open(args.json_path) as f:
        data = json.load(f)

        x_test = [
                [os.path.join(os.path.dirname(args.json_path), c[str(i)]) for i in range(7)]
                 for c in data.get('test')
                 ]
        y_test = [p['label'] for p in data.get('test')]
        aarpid_test = [p['aarp_id'] for p in data.get('test')]
        ts_test = [p['timestamp'] for p in data.get('test')]

    model = get_trained_model(args.modelpath)

    count = 0

    # set the transparency value to be used for overlaying mask on actual image
    alpha = 0.6

    temp_dir = tempfile.mkdtemp(dir='./')

    # iterate over each individual sample in the testing set
    combined_list = list(zip(x_test, y_test, aarpid_test, ts_test))
    random.shuffle(combined_list)
    for row, label, aarpid, ts  in combined_list:
        if count > 100:
            break

        if label == 0:
            base_path = os.path.join(temp_dir,"non_flared")
            if not os.path.exists(base_path):
                os.mkdir(base_path)
        else:
            base_path = os.path.join(temp_dir,"flared")
            if not os.path.exists(base_path):
                os.mkdir(base_path)

        #images = tf.data.Dataset.from_generator(generator = lambda: img_generator(row),
        #        output_types = tf.float32,
        #        output_shapes = [7, 512, 512])

        images = _parse_images(row)

        images, images_pre = preprocess_data_E4(images)
        attribution_masks  = get_attributions_mask(images, model, args)
        prediction = model.predict(images)

        ce = np.abs(cross_entropy(label, prediction))

        print("True label", label)
        print("Saving to base path", base_path)
        print("AARP ID", aarpid)
        print("Timestamp", ts)
        print("Cross-entropy loss", ce)

        pdf_path = f"overlay_mask_{ce:.5f}_{count}_{aarpid}_{ts}_{alpha}_all.pdf"

        with PdfPages(os.path.join(base_path,pdf_path)) as pdf:
            for i in range(7):
                attribution_mask = attribution_masks[:,:,i]
                plot_single_channel_attribution(attribution_mask, images_pre, channel=i )
                pdf.savefig()
        #dataset = tf.data.Dataset.from_tensor_slices((images, label))
                plt.close()
        count += 1

