# from train_alexnet import get_compiled_model
import numpy as np
import matplotlib.pyplot as plt
from .integrated_gradients import get_attributions_mask
from .attribution_sequence import attribution_sequence

class aarp_intensities_with_attribution:
    def __init__(self, aarp_sequence, attribution_sequence):
        self.label = aarp_sequence.label
        self.aarp_id = aarp_sequence.aarp_id
        self.all_wavelengths = aarp_sequence.all_wavelengths
        self.aarp_sequence = aarp_sequence
        self.attribution_sequence = attribution_sequence

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
