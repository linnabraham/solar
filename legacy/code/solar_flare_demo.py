from glob import glob
import sys
import os
from solardemo import solardemo

if __name__=="__main__":
    #aia_171_dir = "/home/linn/july/solar/data/1690551846"
    #aia_171_paths = glob(os.path.join(aia_171_dir,"*.fits"))

    #aia_131_dir = "/home/linn/july/solar/data/1690543290"
    aia_131_dir = "/home/linn/july/solar/data/1690534254"
    aia_131_paths = glob(os.path.join(aia_131_dir, "*.fits"))


    #hmi_dir = "/home/linn/july/solar/data/1690906835"
    #hmi_paths = glob(os.path.join(hmi_dir, "*.fits"))

    sf = solardemo()

    #sf.read_hmi(file_paths = hmi_paths)
    #sf.hmi.peek()
    #solardemo.anim_HMI(sf.hmi)
    #plt.show()

    #sf.read_aia(key=171, file_paths = aia_171_paths)
    sf.read_aia(key=131, file_paths = aia_131_paths)
    sf.aia[131] = solardemo.downscale_map(sf.aia[131], dim=[512, 512])
    #solardemo.anim_ims(sf.aia[171], sf.aia[131])

    flux_datapath = '/home/linn/july/solar/data/XRS/sci_gxrs-l2-irrad_g15_d20140107_v0-0-0.nc'
    flare_start = "2014-01-07T18:04:00"
    flare_end = "2014-01-07T18:58:00"

    #sf.read_flare(flux_datapath, flare_start, flare_end)
    #sf.flux = solardemo.resample_flux(mapseq = sf.aia[171], flux = sf.flux)
    #sf.flux = solardemo.resample_flux(mapseq = sf.aia[131], flux = sf.flux)
    #sf.flux['a_flux'].plot()
    #plt.show()

    #solardemo.anim_AIA(aia_l15_maps)
    #solardemo.anim_AIA(sf.aia[131])
    #solardemo.capture_AIA(sf.aia[131])
    aia_l15_maps = solardemo.l1_to_l15(sf.aia[131])
    #solardemo.capture_AIA(aia_l15_maps)
    solardemo.aia_to_png(aia_l15_maps, "aia_flare_512")
    sys.exit(0)
    #solardemo.anim_sync(sf.flux)
    #solardemo.anim_ts_sync(sf.aia[171], sf.flux)
    #solardemo.anim_ts_sync(sf.aia[131], sf.flux, vmax=200)
    #email = os.environ.get('JSOC_EMAIL')
    #sf.download_HMI(flare_start, flare_end, email)
    #solardemo.anim_ts_sync_all(sf.aia[171], sf.aia[131], sf.hmi, sf.flux) 

