#!/bin/env python
"""
A standalone script that accepts the path to a fits 7h cube and returns a padded image
"""
import argparse
from astropy.io import fits
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
import sunpy.visualization.colormaps as cm
sdoaia304 = matplotlib.colormaps['sdoaia304']

def downscale_and_pad(image, target_shape=(512,512)):
    """
    Function to downsize image to specified size
    Resizing is attempted in an aspect ratio aware way
    The aspect ratio is computed and used to fix either the width or height.
    The difference in the other dimension is calculated and this dimension is filled using 
    the quiet sun background by sampling from all the edges of the image 2 pixels wide.
    If the difference is odd, one of the edges is retained as black.

    """
    target_height, target_width = target_shape
    aspect_ratio = image.shape[1]/image.shape[0]
    #assert aspect_ratio != 1.
    if aspect_ratio > 1:
        padded_height = int(target_width/aspect_ratio)
        img_to_paste = np.array(Image.fromarray(image).resize((target_width, padded_height)))
        height_diff = target_height - padded_height
        #assert height_diff%2 == 0
        qs_pixels = list(image[:2, :].flatten()) + list(image[-2:,:].flatten())+ \
                list(image[:,:2].flatten()) + list(image[:,-2:].flatten())
        black = np.zeros((target_height, target_width))
        half_diff = int(height_diff/2)
        black[:half_diff,:] = np.random.choice(qs_pixels, size=(black[:half_diff,:].shape))
        if height_diff % 2 == 1:
            org_height = img_to_paste.shape[0]
            black[half_diff:-half_diff-1,:] = img_to_paste
            black[-half_diff-1:,:] = np.random.choice(qs_pixels, size=(black[-half_diff-1:,:].shape))
        else:

            black[half_diff:-half_diff,:] = img_to_paste
            black[-half_diff:,:] = np.random.choice(qs_pixels, size=(black[-half_diff:,:].shape))
        return black

    elif aspect_ratio < 1:
        padded_width = int(target_height * aspect_ratio)
        img_to_paste = np.array(Image.fromarray(image).resize((padded_width, target_height)))
        width_diff =  target_width - padded_width
        #assert width_diff%2 == 0
        qs_pixels = list(image[:2, :].flatten()) + list(image[-2:,:].flatten())+ \
                list(image[:,:2].flatten()) + list(image[:,-2:].flatten())
        black = np.zeros((target_height, target_width))
        half_diff = int(width_diff/2)
        black[:, :half_diff] = np.random.choice(qs_pixels, size=(black[:, :half_diff].shape))

        if width_diff%2 == 1:
            org_width = img_to_paste.shape[1]
            black[:, half_diff:-half_diff-1] = img_to_paste
            black[:, -half_diff-1:] = np.random.choice(qs_pixels, size=(black[:, -half_diff-1:].shape))
        else:
            black[:, half_diff:-half_diff] = img_to_paste
            black[:, -half_diff:] = np.random.choice(qs_pixels, size=(black[:, -half_diff:].shape))
        return black

    else:
        resized = np.array(Image.fromarray(image).resize((target_width, target_height)))
        return resized


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-data', '--data')
    args = parser.parse_args()
    fits_cube = fits.open(args.data)

    try:
        img_to_pad = fits_cube[4].data[5]
        print("Shape of image to pad", img_to_pad.shape)
        plt.imshow(img_to_pad, cmap=sdoaia304, vmax=200)
        plt.savefig("before_padding.png")
    except Exception as e:
        print(e)

    ret = downscale_and_pad(img_to_pad, target_shape=(512,512))
    plt.imshow(ret, cmap=sdoaia304, vmax=200)
    plt.savefig("padded_im.png")
