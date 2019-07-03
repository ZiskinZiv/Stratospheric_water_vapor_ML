#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 13:29:51 2019

@author: shlomi
"""

from strat_startup import *
import pandas as pd
import geopandas as gpd
igra2 = pd.read_fwf(cwd / 'igra2-station-list.txt', header=None)
igra2.columns = ['station_number', 'lat', 'lon', 'alt', 'name', 'start_year',
                 'end_year', 'number']
igra2 = igra2[igra2['lat'] > -90]
world = gpd.read_file(cwd / 'gis/Countries_WGS84.shp')
geo_igra2 = gpd.GeoDataFrame(igra2, geometry=gpd.points_from_xy(igra2.lon,
                                                                igra2.lat),
                               crs=world.crs)
ax = world.plot()
geo_igra2.plot(ax=ax, column='alt', cmap='Reds', edgecolor='black',
                     legend=True)