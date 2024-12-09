# from train_alexnet import get_compiled_model
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sunpy.visualization.colormaps as cm
import os
from matplotlib.animation import FuncAnimation
from .integrated_gradients import get_attributions_mask
from .attribution_sequence import attribution_sequence

class aarp_intensities_with_attribution:
    def __init__(self, aarp_sequence, attribution_sequence):
        self.label = aarp_sequence.label
        self.aarp_id = aarp_sequence.aarp_id
        self.all_wavelengths = aarp_sequence.all_wavelengths
        self.aarp_sequence = aarp_sequence
        self.attribution_sequence = attribution_sequence

    def make_contour_movie(self, passband, percentile_level, sqrt=False):
        if sqrt:
            intensities_data = self.aarp_sequence.get_images(passband=passband, non_negative=True)
        else:
            intensities_data = self.aarp_sequence.get_images(passband=passband)
        attribution_data = self.attribution_sequence.get_images(passband=passband)
        nframes = intensities_data.shape[0]
        aia_cmap = matplotlib.colormaps[f'sdoaia{passband}']
        fig, ax = plt.subplots()
        if sqrt:
            im = ax.imshow(np.sqrt(intensities_data[0,:,:]), cmap=aia_cmap, origin='lower')
        else:
            im = ax.imshow(intensities_data[0,:,:], cmap=aia_cmap, origin='lower')
        cbar = fig.colorbar(im, ax=ax)
        threshold = np.percentile(attribution_data, percentile_level)
        print("Using threshold:", threshold)
        contour = None
        c_intensities = []
        suffix = ""
        if sqrt:
            suffix+="_sqrt"
        file_path = f"aarp_{self.aarp_id}_passband_{passband}_contour_movie_plevel_{percentile_level}{suffix}.mp4"
        if os.path.exists(file_path):
            raise ValueError(f"{file_path} already exists")
        def update(frame):
            nonlocal contour
            im_masked = np.ma.masked_where(attribution_data[frame,:,:] < threshold, attribution_data[frame,:,:])
            if sqrt:
                im.set_array(np.sqrt(intensities_data[frame,:,:]))
            else:
                im.set_array(intensities_data[frame,:,:])
            cbar.update_normal(im)
            if contour is not None:
                for c in contour.collections:
                    c.remove()
            contour = ax.contour(im_masked, cmap='jet', origin='lower')
            c_intensities.extend(intensities_data[frame,:,:][im_masked < contour.levels[-1]].flatten())
            # print("len of c_intensities", len(c_intensities))
            # print("contour levels", contour.levels)
            ax.set_title(f"{self.aarp_sequence.timestamps[frame]}_AARP_Id:{self.aarp_id}_passband_{passband}")
        ani = FuncAnimation(fig, update, frames =nframes, interval=50)
        ani.save(file_path, writer='ffmpeg', fps=5)
        return c_intensities

class aarp_ig:
    def __init__(self, input_shape, num_channels, model=None):
        self.model = model
        self.input_shape = input_shape
        self.num_channels = num_channels

    def get_attribution_sequence(self, aarp_sequence):
        attribution_seq = attribution_sequence(aarp_sequence)
        attribution_data = []
        for timestamp, image_dict in aarp_sequence.data:
            image = np.stack(list(image_dict.values()), axis=0)
            # image = np.stack(list(image_dict.values()), axis=-1)

            attribution_mask = get_attributions_mask(image,
                                                    self.model,
                                                    aarp_sequence.label,
                                                    self.input_shape,
                                                    self.num_channels)
            attribution_mask_arr = attribution_mask.numpy()
            attribution_mask_dict = {}
            for idx in range(attribution_mask_arr.shape[-1]):

                attribution_mask_dict[aarp_sequence.all_wavelengths[idx]] = \
                        attribution_mask_arr[:,:,idx]
            attribution_data.append((timestamp, attribution_mask_dict))
            attribution_seq.data = attribution_data
        return attribution_seq
