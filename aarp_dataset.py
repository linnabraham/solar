import sys, os
sys.path.append(os.path.expanduser("~/2024/nov/flares"))
import json
import matplotlib
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(42)
from datetime import datetime, timedelta
from dateutil.parser import isoparse
import matplotlib.dates as mdates
import pandas as pd
import seaborn as sns
sns.set_theme()
from aia_utils import read_fits, plot_aia_image, make_aia_movie

def print_image_stats(data, percentile_level=99):
    print("Image shape:", data.shape)
    print("Image minimum:", np.min(data))
    print("Image maximum:", np.max(data))
    print(f"The {percentile_level}th percentile value is: ", np.percentile(data, percentile_level))


def make_log_safe(data):
    min_pos_value = np.min(data[data > 0])
    epsilon = min_pos_value * 1e-5
    data_safe = np.where(data > 0, data, epsilon)
    return data_safe

def log_transform_flatten(data, method='naive'):
    if np.any(data < 0):
        raise ValueError("Data contains negative values")
    if method != 'naive':
        raise NotImplementedError
    else:
        return np.log(data[data>0])

def plot_intensity_distribution(data, ax=None, xlabel=None, **kwargs):
    if ax is None:
        ax = plt.gca()

    data_min = data.min()
    data_max = data.max()
    intensity_hist, intensity_bins  = np.histogram(data, bins=50, range=(data_min, data_max), density=True)
    # plt.figure(figsize=(10,6))
    ax.bar(intensity_bins[:-1], intensity_hist, width=np.diff(intensity_bins), **kwargs)
    if xlabel:
        title = f"Distribution of {xlabel}"
    else:
        title = "Distribution"
    ax.set_title(title)
    ax.set_ylabel('Normalized counts')
    if xlabel:
        plt.xlabel(xlabel)
    ax.grid(True)
    ax.legend()
    return ax

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

    def plot_goes_with_aarp_sampling(self, goes_ts_data):
        timestamps = pd.to_datetime(self.timestamps, utc=True)
        ts_min = timestamps.min()
        ts_max = timestamps.max()
        assert isinstance(ts_min, pd._libs.tslibs.timestamps.Timestamp)

        fig, ax = plt.subplots(figsize=(15,10))
        ax.set_xlim(ts_min, ts_max)

        for ts in timestamps:
            ax.axvline(ts, color='grey', linestyle='--')
        goes_ts_data.plot(columns=['xrsb'])
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.title("GOES Timeseries with AARPS sampling")
        plt.show()

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

    def create_aarp_movie(self, filename, passband, sqrt=True):
        aia_cmap = matplotlib.colormaps[f'sdoaia{passband}']
        data = self.images[:,self.all_wavelengths.index(passband),:,:]
        nframes = data.shape[0]
        fig, ax = plt.subplots()
        assert self.label is not None
        if sqrt:
            data_nn = np.where(data<0, np.zeros_like(data), data)
            frame_0 = np.sqrt(data_nn[0,:,:])
        else:
            frame_0 = data[0,:,:]
        im = ax.imshow(frame_0, cmap = aia_cmap, origin='lower')
        def update(frame):
            im.set_array(np.sqrt(data[frame,:,:]))
            ax.set_title(f'{self.timestamps[frame]}_AARP_Id:{self.aarp_id}_Filter:{passband}_label:{self.label}')
        ani = FuncAnimation(fig, update, frames = nframes, interval=50)
        ani.save(f'{filename}', writer='ffmpeg', fps=5)


class data_subset:
    def __init__(self, json_data, subset_name):
        self.subset_name = subset_name
        self.json_data = json_data
        self._sample_image = None
        self.subset = self.json_data[self.subset_name]
        self.channels = self.json_data.get('channels')
        self.all_wavelengths = [ passband for idx, passband in self.channels.items()]

    @property
    def file_paths(self):
        if self.subset_name is None:
            raise ValueError("subset_name must be set before accessing file_paths.")

        return [{k: v for k, v in subset_dict.items() if k.isdigit()} for subset_dict in self.subset]

    @property
    def timestamps(self):
        if self.subset_name is None:
            raise ValueError("subset_name must be set before accessing file_paths.")
        return [isoparse(datum['timestamp'])  for datum in self.subset]

    @property
    def labels(self):
        return [datum['label'] for datum in self.subset]

    @property
    def unique_aarp_ids(self):
        if self.subset_name is None:
            raise ValueError("subset_name must be set before accessing file_paths.")
        return set(sorted(np.unique(np.array([datum['aarp_id'] for datum in self.subset]))))

    @property
    def sample_image(self):
        return self._sample_image

    @sample_image.setter
    def sample_image(self, value):
        self._sample_image = value

    def  _generate_sample_image(self):
        """
        Internal method to generate a random sample image.
        """
        #TODO: Add validation to make sure subset is already set?
        channel_idx = np.random.choice(list(self.json_data['channels'].keys()))
        file_path = np.random.choice([item[channel_idx] for item in self.file_paths])
        data = read_fits(file_path)
        self.sample_image = {"path": file_path, "data": data}

    def subset_info(self):
        if self.subset_name is None:
            raise ValueError("Subset is not set")
        print("subset name:", self.subset_name)
        print("Length:", len(self.file_paths))
        time_earliest = min(self.timestamps).isoformat()
        time_latest = max(self.timestamps).isoformat()
        print(f"Time range: {time_earliest} - {time_latest}")
        print(f"Unique AARP ids: {sorted(list(self.unique_aarp_ids))}")
        total = len(self.labels)
        flared_num = sum(self.labels)
        print(f"Flared samples:{flared_num}, Non-Flared samples:{total - flared_num} (Imbalance: {(total - flared_num)/flared_num})")

        aarps_ids_labels = [ (p['aarp_id'], p['label']) for p in self.json_data[self.subset_name]]
        aarp_ids = [aarp_id for aarp_id, label in aarps_ids_labels]
        flared_aarp_ids = [ aarp_id for aarp_id, label in aarps_ids_labels if label == 1]
        print(f"Flared AARPs:{set(flared_aarp_ids)}")

        print(f"Sample Image Path:", self.sample_image.get('path') if self.sample_image else None)
        if self.sample_image:
            print(f"Sample Image stats")
            print(f"==================\n")
            print_image_stats(self.sample_image.get('data'))
        print("\n")


    def create_aarp_sequence(self, aarp_id):
        relevant_entries = [entry for entry in self.subset if entry["aarp_id"] == aarp_id]
        label = relevant_entries[0]["label"]  # Assuming all entries have the same label
        aarp_seq = aarp_sequence(aarp_id=aarp_id, label=label, all_wavelengths=self.all_wavelengths)
        for entry in relevant_entries:
            timestamp = entry["timestamp"]
            image_multiband = {}
            # Add observations for each wavelength from the filtered entries
            for idx, fits_path in entry.items():
                if idx.isdigit():  # Check if the key is a digit (to exclude "label", "aarp_id", and "timestamp")
                    im = read_fits(fits_path)
                    image_multiband[self.all_wavelengths[int(idx)]] = im
            aarp_seq.add_image(timestamp, image_multiband)
        return aarp_seq

    def plot_image(self, data_or_file_path, passband, **kwargs):
        if isinstance(data_or_file_path, str):
            file_path = data_or_file_path
            try:
                data = read_fits(file_path)
            except Exception as e:
                print(e)
        # TODO:validate that data is a numpy array?
        plot_aia_image(data, passband, **kwargs)
        self.sample_image = {'path':file_path, 'data':data}

    def plot_random_image(self, passband = None, force=True, **kwargs):
        """
        Plot a random image from the dataset, belonging to the 171 passband.
        The subset_name needs to be set for this function to work.
        If a file_path and passband is given by the user then plot that image.
        """
        if not force == True:
            if self.sample_image is not None:
                file_path = self.sample_image.get('path')
                data = self.sample_image.get('data')

                if passband is None:
                    raise ValueError("Passband has to be specified when sample image already exists")
                plot_aia_image(data, passband, **kwargs)
            else:
                raise ValueError("The sample image attribute has to be set or the force attribute has to be True")
        else:
            if passband is None:
                passband = 171
            channel_idx = str(self.all_wavelengths.index(passband))
            file_path = np.random.choice([item[channel_idx] for item in self.file_paths])
            data = read_fits(file_path)
            plot_aia_image(data, passband, **kwargs)
            self.sample_image = {'path':file_path, 'data':data}
        return file_path, data


class aarp_dataset:
    def __init__(self, json_path):
        with open(json_path) as f:
            data = json.load(f)
        self.json_data = data

    def get_subset(self, subset_name):
        subset = data_subset(self.json_data, subset_name=subset_name)
        return subset

    def show(self):
            for subset in ['training', 'validation', 'test']:
                print(f"{subset} samples:", len(self.json_data.get(subset)))

if __name__=="__main__":
    import sys
    from solar_library import fetch_goes_data
    import pickle
    dataset = aarp_dataset("/home/linn/july/solar/solar_dataset.json")
    # train_ds = dataset.get_subset(subset_name='training')
    # train_ds.subset_info()
    # aarp_seq = train_ds.create_aarp_sequence(aarp_id=1321)
    test_ds = dataset.get_subset(subset_name='test')
    print(test_ds.unique_aarp_ids)
    for aarp_id in test_ds.unique_aarp_ids:
        print("Processing AARP ID:", aarp_id)
        aarp_seq = test_ds.create_aarp_sequence(aarp_id=aarp_id)
        print("Pickling output to disk...")
        with open(f'aarp_seq_{aarp_id}.pkl', 'wb') as file:
            pickle.dump(aarp_seq, file)
    sys.exit(0)
    aarp_seq = test_ds.create_aarp_sequence(aarp_id=3291)
    aarp_seq_171 = aarp_seq.get_images(passband=171)
    print(aarp_seq_171.shape)
    aarp_seq.create_aarp_movie("first_aarp_movie.mp4", passband=171)
    print(aarp_seq.images.shape)
    print(len(aarp_seq.timestamps))
    dt = aarp_seq.timestamps[0]
    #print("Is timezone-aware?", dt.tzinfo is not None)
    import pandas as pd
    timestamps = pd.to_datetime(aarp_seq.timestamps, utc=True)
    dt_min =  min(timestamps)
    dt_max =  min(timestamps)
    dt_min_with_z = '2013-10-19T15:42:01Z'
    dt_max_with_z = '2013-10-26T21:54:01Z'
    dt_min = isoparse(dt_min_with_z)
    dt_max = isoparse(dt_max_with_z)
    print(dt_min, dt_max)
    goes_data_ts  = fetch_goes_data(dt_min, dt_max)
    dt_min_with_z = dt_min.isoformat().replace("+00:00", "Z")
    dt_max_with_z = dt_max.isoformat().replace("+00:00", "Z")

    #print("Is timezone-aware?", dt.tz is not None)

    goes_data_ts  = fetch_goes_data(dt_min_with_z, dt_max_with_z)
    #dataset.show()
    #dataset._generate_sample_image()
    dataset = aarp_dataset("/home/linn/july/solar/solar_dataset.json", subset='training')
    #dataset.show()
    #dataset = aarp_dataset("/home/linn/july/solar/solar_dataset.json", subset='validation')
    #dataset.show()
    #dataset = aarp_dataset("/home/linn/july/solar/solar_dataset.json", subset='test')
    #dataset.show()
    print(dataset.timestamps[0])
    dataset._generate_sample_image()
    im = dataset.sample_image['data']
    ar_patch = aarp_sequence()
    print(ar_patch.data)
    ar_patch.add_image( dataset.timestamps[-1], {"94":im, "131":im})
    ar_patch.add_image( dataset.timestamps[0], {"94":im, "131":im})
    #print(ar_patch.data)
    print(ar_patch.data[0][1].keys())
    print(ar_patch.data[1][1].keys())
    print(ar_patch.images.shape)
    #print(ar_patch.patch_size)
