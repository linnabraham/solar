import pandas as pd
import os
import subprocess
import datetime
from dateutil.relativedelta import relativedelta

"""
The AARPS data is exposed through a http webpage which has subfolders one for each month of the year
starting from 2010-06 and ending in 2018-12
This script is meant to traverse iteratively through each directory and scrape all
the .fits file links to a text file
"""

def get_from_page_gen(url, patt_list=None, ext=".fits"):
    """
    A generator function to find all links with a particular extension present in a webpage that matches a pattern list

    Parameters:
    url: URL of the webpage to scrape
    patt_list: A list of strings any of which should match with the lists of urls extracted
    ext: extension of files we are interested in

    Returns:
    url that matches all the criteria
    """
    from bs4 import BeautifulSoup
    import requests
    import subprocess

    r  = requests.get(url)
    data = r.text
    soup = BeautifulSoup(data)
    for link in soup.find_all('a'):
        name = link.get('href')
        if name.endswith(ext):
            if name.startswith("./"):
                name = name[2:]
                if patt_list is not None:
                    if name not in patt_list:
                        continue
                yield(name)

if __name__=="__main__":

    baseurl="https://umbra.nascom.nasa.gov/contributed/AIA_AARPS/"
    start_date = datetime.date(year=2010, month=6, day=1)
    end_date = datetime.date(year=2018, month=12, day=31)

    current_date = start_date
    while current_date <= end_date:
        month = current_date.month
        year = current_date.year
        comb = str(year) + f"{month:0>2d}"
        page = baseurl + comb
        current_date = current_date + relativedelta(months=1)
        gen = get_from_page_gen(page)
        output_file = 'aarps_full_urlist.txt'
        print("Adding links to file..")
        with open(output_file, 'a') as file:
            for link in gen:
                #print(link)
                fullpath = os.path.join(baseurl,link)
                subprocess.run(['echo', fullpath], stdout=file, text=True)
