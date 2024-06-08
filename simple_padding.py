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

def pad_along_height(image, target_shape=(512,512)):
    qs_pixels = list(image[:2, :].flatten()) + list(image[-2:,:].flatten())+ \
        list(image[:,:2].flatten()) + list(image[:,-2:].flatten())
    aspect_ratio = image.shape[1]/image.shape[0]
    target_height, target_width = target_shape
    height_before_pad = int(target_width/aspect_ratio)
    #print("Height before pad", height_before_pad)
    height_diff = target_height - height_before_pad
    half_diff = height_diff // 2
    #print("Half diff", half_diff)
    black = np.zeros(target_shape)
    target_width = target_shape[1]
    resized_image = np.array(Image.fromarray(image).resize((target_width,
                                                            height_before_pad)))
    #print("shape of resized image", resized_image.shape)
    if height_diff % 2 == 0:
        black[half_diff:-half_diff,:] = resized_image
    else:
        black[half_diff:-(half_diff+1),:] = resized_image
        black[-half_diff-1,:] = resized_image [-1,:]


    black_top = black[:half_diff,:]
    top_qs = np.random.choice(qs_pixels, size=(black_top.shape))
    for row in np.arange(top_qs.shape[0]):
        row_from_bottom = top_qs.shape[0]-row
        top_qs_height = top_qs.shape[0]
        x = row_from_bottom/top_qs_height
        black_top[row_from_bottom-1, :] = top_qs[row_from_bottom-1,:] * (1-x) + \
                resized_image[0,:] * x

    black_bottom = black[-half_diff:,:] 
    bottom_qs = np.random.choice(qs_pixels, size=(black_bottom.shape))
    for row in np.arange(bottom_qs.shape[0]):
        row_from_bottom = bottom_qs.shape[0]-row
        bottom_qs_height = bottom_qs.shape[0]
        x = row_from_bottom/bottom_qs_height
        #black_bottom[-(row_from_bottom-1), :] = bottom_qs[-(row_from_bottom-1),:] * (1-x) + resized_image[-1:,:] * x
        black_bottom[-(row_from_bottom), :] = bottom_qs[-(row_from_bottom),:] * (1-x) + resized_image[-1:,:] * x

    return black

def pad_along_width(image, target_shape=(512,512)):
    qs_pixels = list(image[:2, :].flatten()) + list(image[-2:,:].flatten())+ \
        list(image[:,:2].flatten()) + list(image[:,-2:].flatten())
    aspect_ratio = image.shape[1]/image.shape[0]
    target_height, target_width = target_shape
    width_before_pad = int(target_height * aspect_ratio)
    #print("Width before pad", width_before_pad)
    width_diff = target_width - width_before_pad
    half_diff = width_diff // 2
    #print("Half diff", half_diff)
    black = np.zeros(target_shape)
    target_width = target_shape[1]
    resized_image = np.array(Image.fromarray(image).resize((width_before_pad,
                                                            target_height)))
    #print("shape of resized image", resized_image.shape)
    if width_diff % 2 == 0:
        black[:,half_diff:-half_diff] = resized_image
    else:
        black[:,half_diff:-(half_diff+1)] = resized_image
        black[:,-half_diff-1] = resized_image [:,-1]


    black_left = black[:,:half_diff]
    left_qs = np.random.choice(qs_pixels, size=(black_left.shape))
    for col in np.arange(left_qs.shape[1]):
        col_from_right = left_qs.shape[1]-col
        left_qs_width = left_qs.shape[1]
        x = col_from_right/left_qs_width
        black_left[:,col_from_right-1] = left_qs[:,col_from_right-1] * (1-x) + \
                resized_image[:,0] * x

    black_right = black[:,-half_diff:]
    right_qs = np.random.choice(qs_pixels, size=(black_right.shape))
    for col in np.arange(right_qs.shape[1]):
        col_from_right = right_qs.shape[1]-col
        right_qs_width = right_qs.shape[1]
        x = col_from_right/right_qs_width
        black_right[:,-col_from_right] = right_qs[:,-col_from_right] * (1-x) + \
                resized_image[:,-3] * x

    return black

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
        return pad_along_height(image, target_shape=target_shape)

    elif aspect_ratio < 1:
        return pad_along_width(image, target_shape=target_shape)

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
