import json
import os
import numpy as np
from astropy.io import fits
#import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from astropy.visualization import ImageNormalize, SqrtStretch
import tensorflow as tf
from helpers.alexnet import AlexNet
from tensorflow.keras import backend
import sys

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

def plot_baseline():
    baseline = tf.zeros(shape=(512,512))
    plt.imshow(baseline)
    plt.title("Baseline")
    plt.axis('off')
    plt.show()

def get_imgs_by_filename(filename):
    #if file_name in img_paths for img_paths in x_train :
    #    print("Yes")
    #if any(file_name in img_paths for img_paths in x_train):
    #    print("Yes")
    indices = [index for index, img_paths in enumerate(x_train) if file_name in img_paths]

    if indices:
        #print("Yes")
        #print("Indices:", indices)
        pass
    else:
        print("No such image found")

    #print(x_train[indices[0]])

    images = _parse_images(x_train[indices[0]])
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

def compute_gradients(images, target_class_idx):
  with tf.GradientTape() as tape:
    tape.watch(images)
    probs = model(images)[target_class_idx]
    #probs = tf.nn.softmax(logits, axis=-1)[:, target_class_idx]
  return tape.gradient(probs, images)

def get_trained_model():
    # force channels-first ordering
    backend.set_image_data_format('channels_first')

    model = AlexNet.build(width=512, height=512, depth=7, classes=1, reg=0.0002)
    classification_threshold = 0.5

    METRICS = [
          tf.keras.metrics.Precision(thresholds=classification_threshold,
                                     name='precision'),
          tf.keras.metrics.Recall(thresholds=classification_threshold,
                                  name="recall"),
          tf.keras.metrics.AUC(num_thresholds=100, curve='PR', name='auc_pr'),
    ]

    model.compile(loss="binary_crossentropy", optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), metrics=METRICS)
    model.load_weights("outputs/best_model.h5")
    return model

def display_interpolation():
    fig = plt.figure(figsize=(20, 20))

    i = 0
    for alpha, image in zip(alphas[0::10], interpolated_images[0::10]):
      i += 1
      plt.subplot(1, len(alphas[0::10]), i)
      plt.title(f'alpha: {alpha:.1f}')
      #print(image[0].shape)
      import matplotlib
      import sunpy.visualization.colormaps as cm
      sdoaia94 = matplotlib.colormaps['sdoaia94']#
      plt.imshow(image[2], cmap=sdoaia94)
      #plt.imshow(np.moveaxis(image[:3,:,:],0,2))
      #image = tf.gather(image, indices=[2,3,4], axis=0)
      #print("Newshape", image.shape)
      #print("newshape",image[[2,3,4],:,:].shape)
      #plt.imshow(np.moveaxis(image,0,2))
      #plt.colorbar()
      #print(np.sum(image[0]))
      plt.axis('off')

    plt.tight_layout();

    #plt.show()
def plot_gradients():
    plt.figure(figsize=(10, 4))
    ax1 = plt.subplot(1, 2, 1)
    ax1.plot(alphas, pred_proba)
    ax1.set_title('Target class predicted probability over alpha')
    ax1.set_ylabel('model p(target class)')
    ax1.set_xlabel('alpha')
    ax1.set_ylim([0, 1])

    ax2 = plt.subplot(1, 2, 2)
    # Average across interpolation steps
    average_grads = tf.reduce_mean(path_gradients, axis=[1, 2, 3])
    # Normalize gradients to 0 to 1 scale. E.g. (x - min(x))/(max(x)-min(x))
    average_grads_norm = (average_grads-tf.math.reduce_min(average_grads))/(tf.math.reduce_max(average_grads)-tf.reduce_min(average_grads))
    ax2.plot(alphas, average_grads_norm)
    ax2.set_title('Average pixel gradients (normalized) over alpha')
    ax2.set_ylabel('Average pixel gradients')
    ax2.set_xlabel('alpha')
    ax2.set_ylim([0, 1]);
    plt.show()

def integral_approximation(gradients):
  # riemann_trapezoidal
  grads = (gradients[:-1] + gradients[1:]) / tf.constant(2.0)
  integrated_gradients = tf.math.reduce_mean(grads, axis=0)
  return integrated_gradients

def integrated_gradients(baseline,
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

    gradient_batch = one_batch(baseline, image, alpha_batch, target_class_idx)
    gradient_batches.append(gradient_batch)

  # Concatenate path gradients together row-wise into single tensor.
  total_gradients = tf.concat(gradient_batches, axis=0)

  # Integral approximation through averaging gradients.
  avg_gradients = integral_approximation(gradients=total_gradients)

  # Scale integrated gradients with respect to input.
  integrated_gradients = (image - baseline) * avg_gradients

  return integrated_gradients

@tf.function
def one_batch(baseline, image, alpha_batch, target_class_idx):
    # Generate interpolated inputs between baseline and input.
    interpolated_path_input_batch = interpolate_images(baseline=baseline,
                                                       image=image,
                                                       alphas=alpha_batch)

    # Compute gradients between model outputs and interpolated inputs.
    gradient_batch = compute_gradients(images=interpolated_path_input_batch,
                                       target_class_idx=target_class_idx)
    return gradient_batch

def plot_img_attributions(baseline,
                          image,
                          target_class_idx,
                          m_steps=50,
                          cmap=None,
                          wav = None,
                          vmax = None,
                          overlay_alpha=0.4):

  attributions = integrated_gradients(baseline=baseline,
                                      image=image,
                                      target_class_idx=target_class_idx,
                                      m_steps=m_steps)
  attributions = np.moveaxis(attributions, 0, 2)
  # Sum of the attributions across color channels for visualization.
  # The attribution mask shape is a grayscale image with height and width
  # equal to the original image.
  #attributions = np.sqrt(attributions)
  attribution_mask = tf.reduce_sum(tf.math.abs(attributions), axis=-1)
  #attribution_mask = np.sqrt(attribution_mask)

  fig, axs = plt.subplots(nrows=2, ncols=2, squeeze=False, figsize=(8, 8))

  axs[0, 0].set_title('Baseline image')
  #axs[0, 0].imshow(baseline[:3, :, :].transpose(1, 2, 0))
  axs[0, 0].imshow(np.moveaxis(baseline[:3, :, :], 0, 2))
  axs[0, 0].axis('off')

  axs[0, 1].set_title('Original image')
  #image = np.moveaxis(image[:3,:,:], 0, 2)
  #image = (image - np.min(image)) / (np.max(image) - np.min(image))
  images = get_imgs_by_filename(file_name)
  image = images[1]

  image = np.moveaxis(images[[5,2,0],:,:],0,2)
  image = (image - np.min(image, axis=(0, 1))) / (np.max(image, axis=(0, 1)) - np.min(image, axis=(0, 1)))
  #image = (image - np.min(image)) / (np.max(image) - np.min(image))
  #import matplotlib
  #import sunpy.visualization.colormaps as cm
  #sdoaia94 = matplotlib.colormaps['sdoaia94']#
  #sdoaia131 = matplotlib.colormaps['sdoaia131']#
  ##import matplotlib.pyplot as plt
  #if wav == "131":
  #    cmap = sdoaia131
  #elif wav == "94":
  #    cmap = sdoaia94
  #if vmax:
  #    axs[0, 1].imshow(image, vmin=0, vmax=vmax, cmap=cmap)
  #    axs[0, 1].axis('off')
  #else:
  #    axs[0, 1].imshow(image, vmin=0, cmap=cmap)
  #    axs[0, 1].axis('off')

  print(np.min(image,axis=(0,1)), np.max(image, axis=(0,1)))
  axs[0, 1].imshow(image, vmin=0.1, vmax=0.3, cmap=cmap)
  axs[0, 1].axis('off')

  axs[1, 0].set_title('Attribution mask')
  axs[1, 0].imshow(attribution_mask, cmap=cmap)
  axs[1, 0].axis('off')

  axs[1, 1].set_title('Overlay')
  axs[1, 1].imshow(attribution_mask, cmap=cmap)
  axs[1, 1].imshow(image, alpha=overlay_alpha)
  axs[1, 1].axis('off')

  plt.tight_layout()
  return fig

def plot_single_channel(images, channel):
    image = images[channel,:,:]
    #image = np.moveaxis(image,0,2)
    image = (image - np.min(image)) / (np.max(image) - np.min(image))
    plt.imshow(image)
    plt.show()

def plot_single(image, wav, vmax=None):
    import matplotlib
    import sunpy.visualization.colormaps as cm
    sdoaia94 = matplotlib.colormaps['sdoaia94']#
    sdoaia131 = matplotlib.colormaps['sdoaia131']#
    import matplotlib.pyplot as plt
    if wav == "131":
        cmap = sdoaia131
    elif wav == "94":
        cmap = sdoaia94
    if vmax:
        plt.imshow(image, vmin=0, vmax=vmax, cmap=cmap)
    else:
        plt.imshow(image, vmin=0, cmap=cmap)



if __name__=="__main__":
    json_path = "solar_dataset.json"
    with open(json_path) as f:
        data = json.load(f)

        x_train = [
                [os.path.join(os.path.dirname(json_path), c[str(i)]) for i in range(7)]
                 for c in data.get('training')
                 ]
        y_train = [p['label'] for p in data.get('training')]
    baseline = tf.zeros(shape=(7,512,512))
    #plot_baseline()
    #plt.imshow(baseline)
    #file_name = "/data/linn/E2/pos_extracted/AARP377_94_2011-02-21T17:48:02Z.fits"
    #file_name = "/data/linn/E2/pos_extracted/AARP377_94_2011-02-10T17:48:02Z.fits"
    #file_name = "/data/linn/E2/pos_extracted/AARP3563_131_2014-01-07T15:48:01Z.fits"
    file_name = "/data/linn/E2/pos_extracted/AARP3563_94_2014-01-07T15:48:01Z.fits"
    #file_name = "/data/linn/E2/pos_extracted/AARP3563_131_2014-01-07T17:48:01Z.fits"
    #file_name = "/data/linn/E2/pos_extracted/AARP3563_94_2014-01-07T21:48:01Z.fits"
    images = get_imgs_by_filename(file_name)
    print("Shape of images", images.shape)
    #plot_single_channel(images, channel=2)

    images = tf.image.per_image_standardization(images)
    print("Shape of images after standardization", images.shape)

    m_steps=50
    alphas = tf.linspace(start=0.0, stop=1.0, num=m_steps+1) # Generate m_steps intervals for integral_approximation() below.
    interpolated_images = interpolate_images(baseline, images, alphas=alphas)
    #display_interpolation()

    model = get_trained_model()
    path_gradients = compute_gradients(
        images=interpolated_images,
        target_class_idx=1)
    print("Shape of path_gradients", path_gradients.shape)
    pred = model(interpolated_images)
    print("Shape of probabilities", pred.shape)
    pred_proba = pred[:,0]
    print("Shape of probabilities", pred_proba.shape)
    #plot_gradients()
    ig = integral_approximation(
        gradients=path_gradients)
    print(ig.shape)
    ig_attributions = integrated_gradients(baseline=baseline,
                                           image=images,
                                           target_class_idx=1,
                                           m_steps=240)
    print("Shape of Integrated gradients", ig_attributions.shape)
    _ = plot_img_attributions(image=images,
                          baseline=baseline,
                          target_class_idx=1,
                          m_steps=240,
                          cmap=plt.cm.inferno,
                          wav="94",
                          vmax=500,
                          overlay_alpha=0.4)
    plt.show()
