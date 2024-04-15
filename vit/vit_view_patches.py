#!/bin/env python
"""
Script to visualize the patches created as part of the ViT model
"""
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras import ops
from tensorflow.keras import layers
import numpy as np
from train_vit import read_fits_cube

class Patches(layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size
    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
                images=images,
                sizes=[1, self.patch_size, self.patch_size, 1],
                strides=[1, self.patch_size, self.patch_size, 1],
                rates=[1, 1, 1, 1],
                padding="VALID",
                )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

def visualize_patches(figname, patches, patch_size):
    n = int(np.sqrt(patches.shape[1]))
    plt.figure(figsize=(8,8))
    for i, patch in enumerate(patches[0]):
        ax = plt.subplot(n, n, i + 1)
        patch_img = tf.reshape(patch, (patch_size, patch_size, 1))
        plt.imshow(patch_img.numpy().astype("uint8"))
        plt.axis("off")
    plt.savefig(figname, bbox_inches="tight")
    plt.show()


if __name__=="__main__":
    patch_size = 36  # Size of the patches to be extract from the input images
    image_size = 512
    num_patches = (image_size // patch_size) ** 2
    projection_dim = 64
    num_heads = 4
    fits_img_path = ["/data/linn/newpipe_extracted_pos/1321_131_2012-01-16T15:42:02Z.fits"]
    image = read_fits_cube(fits_img_path, 512, 512)
    image = np.moveaxis(image, 0, 2)
    image = tf.convert_to_tensor([image])
    patches = Patches(patch_size)(image)
    print(f"Image size: {image_size} X {image_size}")
    print(f"Patch size: {patch_size} X {patch_size}")
    print(f"Patches per image: {patches.shape[1]}")
    print(f"Elements per patch: {patches.shape[-1]}")

    visualize_patches("whatever.png", patches, patch_size)
