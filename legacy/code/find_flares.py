#!/bin/env python
"""
Script for finding flares from the GOES catalogue that matches with the time window selected by AARP dataset authors

Quote from paper:
...Still, 72 s sampling over 6 hr is a large data load. Hence we further downselect to ≈13 minutes of images at the 72 s cadence, centered hourly 15:48–21:48 UT (seven hourly “bursts” of images, over 6 hr, inclusive). The choice of timing is driven by an already-developed set of HMI vector-field time-series extracted data set (Leka et al. 2018).

"""
import sys,os
current_script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_script_dir, ".."))
sys.path.append(parent_dir)
from solar_flare_demo import solardemo
import datetime
import pandas as pd

def every_day(start_date: datetime.date, end_date: datetime.date, events:pd.DataFrame):
    current_date = start_date
    flare_num = 0
    # Iterate over the calendar year using a while loop
    while current_date <= end_date:
        # Increment the current date by one day
        current_date += datetime.timedelta(days=1)    
        desired_time = datetime.time(hour=15, minute=48)
        look_startdatetime = datetime.datetime.combine(current_date, desired_time)
        desired_time = datetime.time(hour=21, minute=48)
        look_enddatetime = datetime.datetime.combine(current_date, desired_time) 
        for _, row in events.iterrows():
            fl_start_datetime = row['start_time'].to_pydatetime()
            fl_end_datetime = row['end_time'].to_pydatetime()
                #TODO: check if endtime is inclusive in AARP dataset
            if fl_start_datetime > look_startdatetime and fl_start_datetime < look_enddatetime:
                if fl_end_datetime < look_enddatetime:
                    flare_num += 1
                    print(flare_num, "Found GOES FLARE:", fl_start_datetime, "->", fl_end_datetime)


if __name__=="__main__":
    sf = solardemo()
    events = solardemo.read_event_list(os.path.join(parent_dir,"data/GOES_event_list.csv"), date_cols = ["event_date","start_time","peak_time","end_time"])
    print("Length of events dataframe", len(events))

    start_date = datetime.date(year=2010, month=6, day=1)
    end_date = datetime.date(year=2018, month=12, day=31)
    every_day(start_date, end_date, events)


