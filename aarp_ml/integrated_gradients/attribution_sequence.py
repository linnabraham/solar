import numpy as np

class attribution_sequence():
    def __init__(self, aarp_sequence):
        self.label = aarp_sequence.label
        self.aarp_id = aarp_sequence.aarp_id
        self.all_wavelengths = aarp_sequence.all_wavelengths
        self.data = []

    @property
    def images(self):
        return self.get_images()

    # TODO: this function is similar to that defined in aarp_dataset class
    def get_images(self, passband=None, non_negative=False, percentile_cutoff=None):
        images_ts = []
        for ts, image_dict in self.data:
            if passband is None:
                image_multiband = [image for image in image_dict.values()]
                images = np.array(image_multiband)
                # images = np.stack(list(image_dict.values()), axis=-1)  # Combine all passbands

            elif passband in image_dict:
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
