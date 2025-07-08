if __name__=="__main__":
    import drms
    c = drms.Client()
    from astropy.io import fits
    from astropy.coordinates import Angle
    import sys
    import sunpy.map
    import requests
    import matplotlib.pylab as plt
    from sunpy.visualization.colormaps import color_tables as ct
    if len(sys.argv) < 2:
        print("Enter HARPNUM")
        print("Enter timestamp in the following format: 2011.02.15_02:12:00_TAI")
        print("Enter keywords")

    harpnum = int(sys.argv[1])
    timestamp = sys.argv[2]
    variables = list([sys.argv[3]])
    print(variables)
    hmi_query_string = f'hmi.sharp_cea_720s[{harpnum}][{timestamp}]'
    print(hmi_query_string)
    email="linna.kkpp@gmail.com"
    r = c.export(hmi_query_string+'{Br}', protocol='fits', email=email)
    fits_url_hmi = r.urls['url'][0]
    print("fits url",fits_url_hmi)
    hmi_map = sunpy.map.Map(fits_url_hmi)
    print("Coordinate frame from SunPy Map",hmi_map.coordinate_frame)
    #print(hmi_map.bottom_left_coord)
    bl_lonval = hmi_map.bottom_left_coord.lon.deg
    bl_latval = hmi_map.bottom_left_coord.lat.deg
    print(bl_lonval, bl_latval)
    tr_lonval = hmi_map.top_right_coord.lon.deg
    tr_latval = hmi_map.top_right_coord.lat.deg
    print(tr_lonval, tr_latval)
    sys.exit(0)
    hmimag = ct.hmi_mag_color_table()
    #plt.imshow(photosphere_image[1].data,cmap=hmimag,origin='lower',vmin=-3000,vmax=3000)
    fig = plt.figure()
    hmi_map.plot(cmap=hmimag, vmin=-3000,vmax=3000)
    plt.show()
    #print('The dimensions of this image are',photosphere_image[1].data.shape[0],'by',photosphere_image[1].data.shape[1],'.')
