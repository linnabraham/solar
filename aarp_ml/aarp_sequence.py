import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from datetime import datetime, timedelta
from dateutil.parser import isoparse
import os
from aarp_ml.utils import run_fetch_goes, plot_goes_with_aarp_sampling

class aarp_sequence:
    def __init__(self, aarp_id=None, label=None, all_wavelengths=None):
        self.label = label
        self.aarp_id = aarp_id
        self.data = []
        self.all_wavelengths = all_wavelengths

    @property
    def images(self):
        return self.get_images()

    def get_images(self, passband=None, non_negative=False, percentile_cutoff=None):
        images_ts = []
        for ts, image_dict in self.data:
            if passband is None:
                image_multiband = [image for image in image_dict.values()]
                images = np.array(image_multiband)
                # image_ts.append(image_multiband)
                # images = np.stack(list(image_dict.values()), axis=-1)  # Combine all passbands

            elif passband in image_dict:
                # images_ts.append(image_dict[passband])
                images = image_dict[passband]
            else:
                raise ValueError(f"Passband {passband} not found in data.")
            # TODO: decide the order of these operation
            if percentile_cutoff:
                threshold = np.percentile(images, percentile_cutoff)
                images = np.where(images < threshold, 0, images)
            if non_negative:
                images = np.where(images < 0, 0, images)

            images_ts.append(images)

        return np.array(images_ts)

    @property
    def timestamps(self):
        return [ ts for ts, image_dict in self.data ]

    def plot_time_sampling(self):
        timestamps = pd.to_datetime(self.timestamps, utc=True)
        ts_min = timestamps.min()
        ts_max = timestamps.max()
        assert isinstance(ts_min, pd._libs.tslibs.timestamps.Timestamp)
        goes_ts_data = run_fetch_goes(ts_min, ts_max)
        plot_goes_with_aarp_sampling(timestamps, goes_ts_data)

    @property
    def patch_size(self):
        if not self.images is None:
            return self.images[0,0,:,:].shape
        else:
            raise ValueError("Images is not set")

    def add_image(self, timestamp, image:dict):
        if isinstance(timestamp, str):
            timestamp = isoparse(timestamp)
        assert isinstance(timestamp, datetime)
        #TODO: do more validation for images
        self.data.append((timestamp, image))
        self.data.sort(key=lambda x:x[0])

    def create_aarp_movie(self, passband, sqrt=False):
        aia_cmap = matplotlib.colormaps[f'sdoaia{passband}']
        data = self.images[:,self.all_wavelengths.index(passband),:,:]
        nframes = data.shape[0]
        fig, ax = plt.subplots()
        assert self.label is not None
        suffix = ""
        if sqrt:
            data_nn = np.where(data<0, np.zeros_like(data), data)
            frame_0 = np.sqrt(data_nn[0,:,:])
            suffix+= "_sqrt"
        else:
            frame_0 = data[0,:,:]

        file_path = f"aarp_{self.aarp_id}_movie_passband_{passband}{suffix}.mp4"
        if os.path.exists(file_path):
            raise ValueError(f"{file_path} already exists")

        im = ax.imshow(frame_0, cmap = aia_cmap, origin='lower')
        def update(frame):
            if sqrt:
                im.set_array(np.sqrt(data[frame,:,:]))
            else:
                im.set_array(data[frame,:,:])
            ax.set_title(f'{self.timestamps[frame]}_AARP_Id:{self.aarp_id}_Filter:{passband}_label:{self.label}{suffix}')
        ani = FuncAnimation(fig, update, frames = nframes, interval=50)
        ani.save(f'{file_path}', writer='ffmpeg', fps=5)
