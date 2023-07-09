# This code takes a sequence of level 1 AIA fits files and processes them to level 1.5 and
# creates a movie out of the files using matplotlib 

from glob import glob
from aiapy.calibrate import correct_degradation, normalize_exposure, register, update_pointing
from aiapy.calibrate.util import get_correction_table, get_pointing_table
import sunpy.map
import matplotlib.pyplot as plt
import astropy.units as u
import matplotlib.animation as animation
import sys
import os

if __name__=="__main__":
    fits_dir = sys.argv[1]
    fits_files = glob(os.path.join(fits_dir,"*.fits"))

    level_1_maps = sunpy.map.Map(fits_files)

    pointing_table = get_pointing_table(level_1_maps[0].date - 3 * u.h, level_1_maps[-1].date + 3 * u.h)
    correction_table = get_correction_table()

    level_15_maps = []

    for a_map in level_1_maps:
        map_updated_pointing = update_pointing(a_map, pointing_table=pointing_table)
        map_registered = register(map_updated_pointing)
        map_degradation = correct_degradation(map_registered, correction_table=correction_table)
        map_normalized = normalize_exposure(map_degradation)
        level_15_maps.append(map_normalized)
    sequence = sunpy.map.Map(level_15_maps, sequence=True)
    ani = sequence.plot()
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=10, metadata=dict(artist='SunPy'), bitrate=1800)
    ani.save('mapsequence_animation.mp4', writer=writer)   
    #sequence.peek()
    #plt.show()
