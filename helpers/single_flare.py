from sunpy.time import TimeRange
from sunkit_instruments import goes_xrs
import pandas as pd
from time import time
from sunpy.net import Fido
from sunpy.net import attrs as a
import drms
from pandas._libs.tslibs.timestamps import Timestamp
import sunpy.map
import matplotlib.animation as animation
from aiapy.calibrate import correct_degradation, normalize_exposure, register, update_pointing
from aiapy.calibrate.util import get_correction_table, get_pointing_table
import astropy.units as u

class SolarFlare:

    @staticmethod
    def get_GOES_EVENTS(t_start="2010-06-01", t_end="2018-12-31"):
        #self.data_start = t_start
        #self.data_end = t_end

        st = time()
        # Grab all the data from the GOES database
        time_range = TimeRange(t_start, t_end)
        # Get only flares of class M1 or above
        listofresults = goes_xrs.get_goes_event_list(time_range, 'M1')
        print('Grabbed all the GOES data; there are', len(listofresults), 'events.')
        print(f'Time taken for download: {time()-st:.2f} seconds')

        df = pd.DataFrame(listofresults)
        df.to_csv("data/GOES_event_list.csv",index=False)

    def get_flare_ts(self, t_start, t_end, sn, download=False):

        #TODO: make this customisable by user
        if sn:
            results = Fido.search(a.Time(t_start, t_end), a.Instrument("XRS"), a.goes.SatelliteNumber(sn))
        else:
            results = Fido.search(a.Time(t_start, t_end), a.Instrument("XRS"))

        if download:
            downloaded_files = Fido.fetch(results, path='../data/{instrument}/{file}')
            self.flare_start = t_start
            self.flare_end = t_end
            return downloaded_files
        else:
            return results

    def get_AIA(self, t_start: Timestamp , t_end: Timestamp, wavelength: int, email, exposure=2.0, download=False):

        t_start = t_start.to_pydatetime().isoformat()
        t_end = t_end.to_pydatetime().isoformat()
        client = drms.Client(email=email)
        keys = ["EXPTIME", "QUALITY", "T_OBS", "T_REC", "WAVELNTH"]
        qstr = f"aia.lev1_euv_12s[{t_start}Z-{t_end}Z][? EXPTIME<{exposure} AND WAVELNTH={wavelength} ?]{{image}}"
        print(f"Querying data -> {qstr}")

        records, filenames = client.query(qstr, key=keys, seg="image")
        print(f"{len(records)} records retrieved. \n")
        print(records)
        #print(filenames)


        if download:
            export = client.export(qstr, method="url", protocol="fits")
            import os
            from time import time
            dirname = int(time())
            os.makedirs(f"data/{dirname}")
            self.aia_files = export.download(f"data/{dirname}")
        else:
            return records, filenames

    @staticmethod
    def make_movie(fits_files: list, dest):
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

        #sequence = sunpy.map.Map(fits_files, sequence=True)
        sequence = sunpy.map.Map(level_15_maps, sequence=True)
        ani = sequence.plot()
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=10, metadata=dict(artist='SunPy'), bitrate=1800)
        ani.save(dest, writer=writer)   
        return sequence

