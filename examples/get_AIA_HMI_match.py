from get_HMI_single import HMIQuery
import astropy.units as u
from astropy.time import Time
from sunpy.net import Fido
from sunpy.net import attrs as a

from sunpy.coordinates import frames

hmi_query = HMIQuery(377, "2011.02.15_02:12:00_TAI", ['QUALITY','HARPNUM','T_REC','CRLN_OBS','CRLT_OBS','CRPIX1','CRPIX2'])
hmi_map = hmi_query.run_query()

bl_hp = hmi_map.bottom_left_coord.transform_to(frames.Helioprojective)
print(bl_hp)
tr_hp = hmi_map.top_right_coord.transform_to(frames.Helioprojective)
print(tr_hp)

start = "2011-02-15T02:10:12.300"
start_time = Time(start, scale='utc', format='isot')
#bl = (-500,-275)
#tr = (150,375)

cutout = a.jsoc.Cutout(bl_hp, top_right=tr_hp, tracking=True)

query = Fido.search(
    a.Time(start_time - 6*u.h, start_time + 6*u.h),
    a.Wavelength(171*u.angstrom),
    a.Sample(2*u.h),
    a.jsoc.Series.aia_lev1_euv_12s,
    a.jsoc.Notify(hmi_query.email),
    a.jsoc.Segment.image,
    cutout,
)
print(query)

files = Fido.fetch(query, path = "./AIA/")
