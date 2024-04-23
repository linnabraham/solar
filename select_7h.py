#!/bin/env python

import pandas as pd
import numpy as np

if __name__=="__main__":

    table_path = "table_data_shapes.csv"
    df = pd.read_csv(table_path, index_col=0)

    df.min_lon = df.min_lon.replace(-999999, np.nan)
    df.max_lon = df.max_lon.replace(-999999, np.nan)
    df = df[~df.min_lon.isna()]
    #print(df.max_lon.isna().sum())
    df = df[df.wavelength != 1600]
    df = df[(np.abs(df.min_lon) < 60) & (np.abs(df.max_lon) < 60)]

    print(df.shape)
    df.to_csv("data/selected_7h.csv", index=False)


