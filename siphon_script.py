#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 13:55:37 2019

@author: ziskin
"""

from pathlib import Path
from strato_soundings import siphon_igra2_to_xarray
sound_path = Path('/home/ziskin/Work_Files/Chaim_Stratosphere_Data/sounding')
cwd = Path().cwd()
import pandas as pd
stations = pd.read_csv(cwd/ 'igra_eq_stations.txt', index_col=0)
for station in stations.values:
    st = station[0]
    siphon_igra2_to_xarray(st, path=sound_path)
