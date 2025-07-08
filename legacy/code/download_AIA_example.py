import sunpy.map
import sunpy.data.sample
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time
from sunpy.net import Fido
from sunpy.net import attrs as a

aia = sunpy.map.Map(sunpy.data.sample.AIA_171_IMAGE)

# Reproduce examples from SunPy page
start = "2012-09-24T14:56:03"
start_time = Time(start, scale='utc', format='isot')
bl = (-500,-275)
tr = (150,375)
#bottom_left = SkyCoord(-500*u.arcsec, -275*u.arcsec, obstime=start_time, observer="earth", frame="helioprojective")
bottom_left = SkyCoord(-500*u.arcsec, -275*u.arcsec,  frame=aia.coordinate_frame)
#top_right = SkyCoord(150*u.arcsec, 375*u.arcsec, obstime=start_time, observer="earth", frame="helioprojective")
top_right = SkyCoord(150*u.arcsec, 375*u.arcsec, frame=aia.coordinate_frame)

cutout = a.jsoc.Cutout(bottom_left, top_right=top_right, tracking=True)

jsoc_email = "linna.kkpp@gmail.com"
query = Fido.search(
    a.Time(start_time - 6*u.h, start_time + 6*u.h),
    a.Wavelength(171*u.angstrom),
    a.Sample(2*u.h),
    a.jsoc.Series.aia_lev1_euv_12s,
    a.jsoc.Notify(jsoc_email),
    a.jsoc.Segment.image,
    cutout,
)
print(query)

files = Fido.fetch(query, path = "./AIA/")
