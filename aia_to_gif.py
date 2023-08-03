# This script read all of the AIA images that were taken during the flare
# normalises the images based on the exposure times and displays an animation
# using sunpy.map.Map with intensity normalization and a square root transformation
# of the data
if __name__=="__main__":
    from solar_flare_demo import solardemo
    import os
    from glob import glob
    import sys

    aia_131_dir = "/home/linn/july/solar/data/1690534254"
    aia_131_paths = glob(os.path.join(aia_131_dir, "*.fits"))

    sf = solardemo()

    sf.read_aia(key=131, file_paths = aia_131_paths)

    aia_l15_maps = solardemo.l1_to_l15(sf.aia[131])
    solardemo.anim_AIA(aia_l15_maps)
