#!/bin/env python

import pandas as pd
import matplotlib.pyplot as plt

import matplotlib.backends.backend_pdf as pdf
pdf_filename = 'outputs/newimages.pdf'
pdf_pages = pdf.PdfPages(pdf_filename)

df = pd.read_csv("data/GOES_event_list.csv")
# print(df.columns)

main_class = df['goes_class'].apply(lambda x: x[0])
plt.hist(main_class)
pdf_pages.savefig()
# plt.show()

print("Number of X class flares", len(df[main_class=='X']))
print("> Number of unique NOAA ARs", len(pd.unique(df['noaa_active_region'][main_class=='X'])))

noaa_ars = [str(item) for item in  pd.unique(df['noaa_active_region'][main_class=='X'])]
noaa_to_harps = pd.read_csv("data/all_harps_with_noaa_ars.txt", delim_whitespace=True)
lens = [ (noaa_to_harps['NOAA_ARS']==str(item)).sum() for item in noaa_ars]
print(lens)

harps = noaa_to_harps.apply(lambda x: x['HARPNUM'] if x['NOAA_ARS'] in noaa_ars else -1, axis=1)
# print(pd.unique(harps)
selected_harps = [ str(harp) for harp in harps if harp > 0]
print(selected_harps)

print(noaa_to_harps.columns)

urldf = pd.read_csv("AARPS_list.txt", header=None, names=['url'])
print("Total no. of urls", len(urldf))

matched = urldf['url'][urldf['url'].str.contains('|'.join(selected_harps), regex=True)]
print("No. of matches in the url list", len(matched))
#matched.to_csv("aarps_list_20230927.csv",index=False, header=None)
print(urldf)

urldf[['Datetime', 'AARP', 'Wavelength']] = urldf['url'].str.extract(r'(\d{4}\.\d{2}\.\d{2}_\d{2}:\d{2}:\d{2})_7h@1h_AARP(\d+)_(\d+)\.fits')

grouped = urldf.groupby(['Datetime', 'AARP'])

# Iterate through each group and check for missing wavelengths
for group_key, group_df in grouped:
    expected_wavelengths = [str(wavelength) for wavelength in (94, 131, 171, 193, 211, 304, 335, 1600)]  # Expected wavelengths
    missing_wavelengths = list(set(expected_wavelengths) - set(group_df['Wavelength']))
    
    if missing_wavelengths:
        print(f"Missing wavelengths for Datetime {group_key[0]} and AARP {group_key[1]}:")
        print(", ".join(missing_wavelengths))

pdf_pages.close()
