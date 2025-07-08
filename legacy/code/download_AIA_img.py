import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time
from sunpy.net import Fido
from sunpy.net import attrs as a

if __name__=="__main__":
    import sys
    if len(sys.argv)<2:
        print("Enter arguments in following order: TIMESTAMP ")
        print("TIMESTAMP: 2012-09-24T14:56:03")
    start = sys.argv[1]
    start_time = Time(start, scale='utc', format='isot')
    bottom_left = sys.argv[2].split(",")
    bl_x_str, bl_y_str = bottom_left
    bl_x = float(bl_x_str)
    bl_y = float(bl_y_str)
    print(bl_x, bl_y)
    top_right = sys.argv[3].split(",")
    tr_x_str , tr_y_str = top_right
    tr_x = float(tr_x_str)
    tr_y = float(tr_y_str)
    print(tr_x, tr_y)
    bottom_left = SkyCoord(bl_x*u.degree, bl_y*u.degree, obstime=start_time, observer="earth", frame="helioprojective")
    top_right = SkyCoord(tr_x*u.degree, tr_y*u.degree, obstime=start_time, observer="earth", frame="helioprojective")
    cutout = a.jsoc.Cutout(bottom_left, top_right=top_right, tracking=True)
    jsoc_email = "linna.kkpp@gmail.com"
    query = Fido.search(
        a.Time(start_time - 0*u.h, start_time + 0*u.h),
        a.Wavelength(171*u.angstrom),
        a.Sample(2*u.h),
        a.jsoc.Series.aia_lev1_euv_12s,
        a.jsoc.Notify(jsoc_email),
        a.jsoc.Segment.image,
        cutout,
    )
    print(query)
    results = Fido.fetch(query)
    downloaded_files = Fido.fetch(results, path='./my_AIA.fits')
    print(downloaded_files)
