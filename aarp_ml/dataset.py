import json
from dateutil.parser import isoparse
import numpy as np
np.random.seed(42)
import matplotlib.pyplot as plt
from .aarp_sequence import aarp_sequence
import sys, os
sys.path.append(os.path.expanduser("~/2024/nov/flares"))
from aia_utils import read_fits, plot_aia_image

def print_image_stats(data, percentile_level=99):
    print("Image shape:", data.shape)
    print("Image minimum:", np.min(data))
    print("Image maximum:", np.max(data))
    print(f"The {percentile_level}th percentile value is: ", np.percentile(data, percentile_level))

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

    def get_file_paths(self, passband=None):
        if passband is None:
            return self.file_paths
        else:
            channel_idx = str(self.all_wavelengths.index(passband))
            return [subset_dict[channel_idx] for subset_dict in self.subset]

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
