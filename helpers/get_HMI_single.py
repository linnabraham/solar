import drms
from astropy.io import fits
from astropy.coordinates import Angle
import sys
import sunpy.map
import requests
import matplotlib.pylab as plt
from sunpy.visualization.colormaps import color_tables as ct

class HMIQuery:
    def __init__(self, harpnum, timestamp, variables):
        self.harpnum = harpnum
        self.timestamp = timestamp
        self.variables = variables
        self.client = drms.Client()
        self.email = "linna.kkpp@gmail.com"
    
    def run_query(self):
        hmi_query_string = f'hmi.sharp_cea_720s[{self.harpnum}][{self.timestamp}]'
        r = self.client.export(hmi_query_string + '{Br}', protocol='fits', email=self.email)
        fits_url_hmi = r.urls['url'][0]
        hmi_map = sunpy.map.Map(fits_url_hmi)
        return hmi_map
    
    def plot_hmi_map(self, hmi_map):
        hmimag = ct.hmi_mag_color_table()
        fig = plt.figure()
        hmi_map.plot(cmap=hmimag, vmin=-3000, vmax=3000)
        plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Enter HARPNUM")
        print("Enter timestamp in the following format: 2011.02.15_02:12:00_TAI")
        print("Enter keywords")
        sys.exit(1)

    harpnum = int(sys.argv[1])
    timestamp = sys.argv[2]
    variables = [sys.argv[3]]
    print(variables)

    hmi_query = HMIQuery(harpnum, timestamp, variables)
    hmi_map = hmi_query.run_query()
    hmi_query.plot_hmi_map(hmi_map)

