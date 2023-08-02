from sunpy.time import TimeRange
from sunkit_instruments import goes_xrs
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import sunpy.map
from astropy.visualization import ImageNormalize, SqrtStretch
import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter
from celluloid import Camera
import drms
from time import time
import os
from aiapy.calibrate import correct_degradation, normalize_exposure, register, update_pointing

class solardemo:
    def __init__(self, event_list_path = None, date_cols = None):

        if event_list_path and date_cols:
            self.event_list = read_event_list(csv_path = event_list_path, date_cols = date_cols)

        self.aia = { 131:[], 171:[], 304:[] }
        self.hmi = []

    def init_flare_times(self, flare_start = None, flare_end=None):
        self.flare_start = flare_start
        self.flare_end = flare_end
    
    @staticmethod
    def read_event_list(csv_path, date_cols: list):
        """
        Inputs:
        csv_path : path to GOES event list in csv format
        date_cols : list of column names in the dataframe to be parsed as dates

        Returns:
        df : pandas dataframe containing GOES event list
        """

        df = pd.read_csv(csv_path, parse_dates=date_cols)
        return df
    
    def read_flare(self, flare_path, flare_start, flare_end):
        """
        Function to read the x-ray flux timeseries from disk and slice it using 
        the flare start and end times provided  

        Inputs:
        flare_start : Flare start time in ISO format
        flare_end : Flare end time in ISO format
        flare_path : path to netCDF file containing x-ray flux in str format

        Returns:
        flux : the sliced timeseries of datatype: xarray.core.dataset.Dataset

        """
        self.init_flare_times(flare_start, flare_end)
        flux  = xr.open_dataset(flare_path)
        self.flux = flux.sel(time=slice(self.flare_start, self.flare_end))

    def read_aia(self, key, file_paths = []):
        self.aia[key] = sunpy.map.Map(file_paths, sequence=True)

    def read_hmi(self, file_paths):
        self.hmi = sunpy.map.Map(file_paths, sequence=True)

    @staticmethod
    def download_GOES_events(t_start="2010-06-01", t_end="2018-12-31", dest="data/GOES_event_list.csv"):

        # Grab all the data from the GOES database
        time_range = TimeRange(t_start, t_end)
        # Get only flares of class M1 or above
        listofresults = goes_xrs.get_goes_event_list(time_range, 'M1')
        print('Grabbed all the GOES data; there are', len(listofresults), 'events.')
        print(f'Time taken for download: {time()-st:.2f} seconds')

        df = pd.DataFrame(listofresults)
        df.to_csv(dest, index=False)

    @staticmethod
    def resample_flux(mapseq : sunpy.map.Map, flux : xr.core.dataset.Dataset):
        """
        Function to resample x-ray flux based on the timestamps in a Sunpy.map sequence

        """
        t_obs = [smap.meta['t_obs'] for smap in mapseq]
        flux = flux.sel(time=t_obs, method='nearest')
        return flux


    @staticmethod
    def anim_ims(map1, map2):
        ims1 = []
        ims2 = []
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 2, 1, projection=map1.maps[0])
        ax2 = fig.add_subplot(1, 2, 2, projection=map2.maps[0])

        #fig, (ax1, ax2) = plt.subplots(1, 2, projection=map1.maps[0])
        ani = map1.plot(axes=ax1, norm=ImageNormalize(vmin=0, vmax=200, stretch=SqrtStretch()))
        ani2 = map2.plot(axes=ax2, norm=ImageNormalize(vmin=0, vmax=200, stretch=SqrtStretch()))
        plt.tight_layout()
        plt.show()

    @staticmethod
    def anim_sync(flux):
        fig, ax = plt.subplots(1, 1)

        xmin = flux['a_flux'].time[0].values
        xmax = flux['a_flux'].time[-1].values

        ymin = min(flux['a_flux'])
        ymax = max(flux['a_flux'])

        def animate(i):
            ax.cla()
            flux['a_flux'][:i].plot()

            ax.set_xlim([xmin, xmax])
            ax.set_ylim([ymin, ymax])
        #plt.tight_layout()

        anim = animation.FuncAnimation(fig, animate, frames = len(flux['a_flux']) + 1, interval = 1, blit=False)
        plt.show()
    @staticmethod
    def anim_ts_sync(map1, flux, vmax=500):


        map1_lists = []

        for amap in map1_list:
            map1.append(normalize_exposure(amap))
        map1 = sunpy.map.Map(map1_lists, sequence=True)

        print("Length of time series", len(flux['a_flux']))
        print("Length of map sequence", len(map1))
        xmin = flux['a_flux'].time[0].values
        xmax = flux['a_flux'].time[-1].values
        print(xmin, map1[0].meta['t_obs'])
        print(xmax, map1[-1].meta['t_obs'])

        writer = FFMpegWriter(fps=5)

        import sys
        #sys.exit(0)

        fig = plt.figure()
        ax1 = fig.add_subplot(1, 2, 1, projection=map1.maps[0])
        ax2 = fig.add_subplot(1, 2, 2)


        ymin = min(flux['a_flux'].values)
        ymax = max(flux['a_flux'].values)

        
        def animate(i):
            ax2.cla()
            #ax1.cla()
            fig = flux['a_flux'][:i].plot(ax=ax2)
            # the following line returns an object of type matplotlib.image.AxesImage
            map1[i].plot(axes=ax1, norm=ImageNormalize(vmin=0, vmax=vmax, stretch=SqrtStretch()))
            #print(type(fig2), "dtype for fig2")

            ax2.set_xlim([xmin, xmax])
            ax2.set_ylim([ymin, ymax])

        #anim1 = map1.plot(axes=ax1, interval=1, resample=[0.25,0.25], norm=ImageNormalize(vmin=0, vmax=500, stretch=SqrtStretch()))
        anim2 = animation.FuncAnimation(fig, animate, frames = len(flux['a_flux']) + 1, interval = 1, blit=False)
        
        #anim2.save("solar_flare_anim.mp4", writer=writer)
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def anim_HMI(map_hmi):
        anim = map_hmi.plot(norm=ImageNormalize(vmin=-1500, vmax=1500), cmap='hmimag')
        plt.show()


    @staticmethod
    def anim_ts_sync_all(map1, map2, map3, flux):
        xmin = flux['a_flux'].time[0].values
        xmax = flux['a_flux'].time[-1].values

        fig = plt.figure()
        ax1 = fig.add_subplot(2, 2, 1, projection=map1.maps[0])
        ax2 = fig.add_subplot(2, 2, 2)
        ax3 = fig.add_subplot(2, 2, 3, projection=map2.maps[0])
        ax4 = fig.add_subplot(2, 2, 4, projection=map3.maps[0])

        ymin = min(flux['a_flux'].values)
        ymax = max(flux['a_flux'].values)

        def animate(i):
            ax2.cla()
            fig = flux['a_flux'][:i].plot(ax=ax2)
            map1[i].plot(axes=ax1,  norm=ImageNormalize(vmin=0, vmax=500, stretch=SqrtStretch()))
            ax2.set_xlim([xmin, xmax])
            ax2.set_ylim([ymin, ymax])
            map2[i].plot(axes=ax3,  norm=ImageNormalize(vmin=0, vmax=500, stretch=SqrtStretch()))
            map3[i].plot(axes=ax4,  norm=ImageNormalize(vmin=-1500, vmax=1500), cmap='hmimag')

        anim2 = animation.FuncAnimation(fig, animate, frames= len(flux['a_flux'])+1, interval=1, blit=False)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def download_HMI(t_start, t_end, email, download=False):
        client = drms.Client(email=email)
        keys = ["QUALITY", "T_OBS", "T_REC" ]

        qstr = f"hmi.M_45s[{t_start}Z-{t_end}Z]{{magnetogram}}"
        print(f"Querying data -> {qstr}")
        records, filenames = client.query(qstr, key=keys, seg="magnetogram")
        print(records)

        if download:
            export = client.export(qstr, method="url",  protocol="fits")
            dirname = f"data/{int(time())}"
            os.makedirs(dirname)
            print("Files are downloaded to ", dirname)
            downloaded_files = export.download(dirname)
        else:
            return records, filenames

    @staticmethod
    def download_data(qstr, email, dest=None):
        client = drms.Client(email=email)
        export = client.export(qstr, method="url", protocol="fits")

        # create a unique dirname using the timestamp of download
        if not dest:
            dirname = f"data/{int(time())}"
        os.makedirs(dirname)
        print("Files are downloaded to :", dirname)
        downloaded_files = export.download(dirname)

    @staticmethod
    def query_AIA(t_start, t_end, wavelength:int, email, exposure=None):
        """
        Function to query JSOC for AIA images with optional filters

        Parameters:
        start and end timestamps
        wavelength
        email: JSOC email
        exposure(optional)

        Returns:
        The query string used
        """

        client = drms.Client(email=email)
        keys = ["EXPTIME", "QUALITY", "T_OBS", "T_REC", "WAVELNTH"]

        qstr = f"aia.lev1_euv_12s[{t_start}Z-{t_end}Z][? WAVELNTH={wavelength} ?]{{image}}"

        if exposure:
            qstr = f"aia.lev1_euv_12s[{t_start}Z-{t_end}Z][? EXPTIME<{exposure} AND WAVELNTH={wavelength} ?]{{image}}"
        print(f"Querying data -> {qstr}")

        records, filenames = client.query(qstr, key=keys, seg="image")
        print(f"{len(records)} records retrieved. \n")
        print(records)
        return qstr



if __name__=="__main__":
    from glob import glob
    import sys

    aia_171_dir = "/home/linn/july/solar/data/1690551846"
    aia_171_paths = glob(os.path.join(aia_171_dir,"*.fits"))

    aia_131_dir = "/home/linn/july/solar/data/1690543290"
    aia_131_paths = glob(os.path.join(aia_131_dir, "*.fits"))

    #aia_maps = sunpy.map.Map(aia_131_paths, sequence=True)
    #aia_l5 = []
    #for amap in aia_maps:
    #    aia_l5.append(normalize_exposure(amap))
    #aia_l5_maps = sunpy.map.Map(aia_l5, sequence=True)
    #aia_l5_maps.peek(norm=ImageNormalize(vmin=0, vmax=500, stretch=SqrtStretch()))
    #plt.show()

    hmi_dir = "/home/linn/july/solar/data/1690906835"
    hmi_paths = glob(os.path.join(hmi_dir, "*.fits"))

    sf = solardemo()

    sf.read_hmi(file_paths = hmi_paths)
    #sf.hmi.peek()
    #solardemo.anim_HMI(sf.hmi)
    #plt.show()

    sf.read_aia(key=171, file_paths = aia_171_paths)
    sf.read_aia(key=131, file_paths = aia_131_paths)

    #solardemo.anim_ims(sf.aia[171], sf.aia[131])

    flux_datapath = '/home/linn/july/data/XRS/sci_gxrs-l2-irrad_g15_d20140107_v0-0-0.nc'
    flare_start = "2014-01-07T18:04:00"
    flare_end = "2014-01-07T18:58:00"

    sf.read_flare(flux_datapath, flare_start, flare_end)
    #sf.flux = solardemo.resample_flux(mapseq = sf.aia[171], flux = sf.flux)
    sf.flux = solardemo.resample_flux(mapseq = sf.aia[131], flux = sf.flux)
    #sf.flux['a_flux'].plot()
    #plt.show()
    #solardemo.anim_sync(sf.flux)
    #solardemo.anim_ts_sync(sf.aia[171], sf.flux)
    solardemo.anim_ts_sync(sf.aia[131], sf.flux, vmax=200)
    email = os.environ.get('JSOC_EMAIL')
    #sf.download_HMI(flare_start, flare_end, email)
    #solardemo.anim_ts_sync_all(sf.aia[171], sf.aia[131], sf.hmi, sf.flux) 

