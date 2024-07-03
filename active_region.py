import matplotlib
import sunpy.visualization.colormaps as cm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import List, Optional, Generator
import numpy as np
from tf_utils import read_fits

class active_region:
    def __init__(self, aarp_id, label):
        self.aarp_id = aarp_id
        self.channels = {
        "0": 94,
        "1": 131,
        "2": 171,
        "3": 193,
        "4": 211,
        "5": 304,
        "6": 335
    }
        self.label = label
        self.data = {}
        self.patch_size = (512,512)

    def add_observation(self, channel, timestamp, fits_path):
        if timestamp not in self.data:
            self.data[timestamp] = {}

        assert isinstance(channel, str)
        wavelength = self.channels[channel]
        if wavelength not in self.data[timestamp]:
            fits_data = read_fits(fits_path)
            assert fits_data.shape ==  self.patch_size
            self.data[timestamp][wavelength] = fits_data

    #def get_observation(self, channel, timestamp):
    #    assert isinstance(channel, str)
    #    wavelength = self.channels[channel]
    #    if timestamp in self.data and wavelength in self.data[timestamp]:
    #        return self.data[timestamp][wavelength]
    #    else:
    #        return None

    def get_observation(self, wavelength):
        observations = []
        #timestamps = []
        if wavelength in self.channels.values():
            timestamps = sorted(
                timestamp for timestamp in self.data.keys() if wavelength in self.data[timestamp])
            for timestamp in timestamps:
                if wavelength in self.data[timestamp]:
                    observations.append(self.data[timestamp][wavelength])
                    #timestamps.append(timestamp)
        return observations, timestamps

    def _get_common_timestamps(self, wavelengths: List[str]) -> List[str]:
        common_timestamps = []
        for timestamp in self.data.keys():
            common_timestamps.append(timestamp)
        return sorted(common_timestamps)

    def _get_observation_generator(self, wavelengths: List[str]) -> Generator[List[Optional[object]], None, None]:
        timestamps = self._get_common_timestamps(wavelengths)
        for timestamp in timestamps:
            observation_for_timestamp = []
            for wavelength in wavelengths:
                observation_for_timestamp.append(self.data.get(timestamp, {}).get(wavelength))
            yield (observation_for_timestamp, timestamp)

    @staticmethod
    def make_aia_movie(filename, data:np.ndarray, wavelength, timestamps:list = None, vmin=None, vmax=None, aarp_id=None, label=None):

        cmap_key = 'sdoaia'+str(wavelength)
        sdoaia_cmap = matplotlib.colormaps[cmap_key]
        nframes = data.shape[0]
        fig, ax = plt.subplots()
        data = np.where(data<0, np.zeros_like(data), data)
        im = ax.imshow(np.sqrt(data[0,:,:]), cmap = sdoaia_cmap, origin='lower')

        def update(frame):
            im.set_array(np.sqrt(data[frame,:,:]))
            if timestamps:
                #print("Found timestamps")
                if aarp_id:
                    if label:
                        ax.set_title(f'{timestamps[frame]}_AARP_Id:{aarp_id}_Filter:{wavelength}_label:{label}')
        ani = FuncAnimation(fig, update, frames = nframes, interval=50)
        ani.save(f'{filename}', writer='ffmpeg', fps=1)

