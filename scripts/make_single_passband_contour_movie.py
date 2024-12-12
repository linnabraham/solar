import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pickle
from aarp_ml.integrated_gradients import aarp_intensities_with_attribution

if __name__=="__main__":
    path_to_intensities_with_attribution = os.path.expanduser("~/july/solar/intensities_with_attributions_7304.pkl")
    with open(path_to_intensities_with_attribution, "rb") as file:
        intensities_with_attributions_seq = pickle.load(file)
    aarp_id = intensities_with_attributions_seq.aarp_id
    print(aarp_id)
    passband = 171
    intensities_with_attributions_seq.make_contour_movie(passband=passband, percentile_level=90, sqrt=False)
