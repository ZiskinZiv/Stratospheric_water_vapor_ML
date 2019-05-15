#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 15 13:41:52 2019

@author: shlomi
"""

import xarray as xr
from dask.diagnostics import ProgressBar
print('Opening large dataset using dask + xarray:')
T = xr.open_mfdataset('/data11/ziskin/ERA5/' + 'era5_T_4*.nc')
T = T.sel(latitude=slice(15, -15))
T = T.sel(level=slice(50, 150))
T = T.rename({'longitude': 'lon', 'latitude': 'lat'})
T = T.sortby('lat')
#     T = T.sortby('lat')

# 1)open mfdataset the temperature data
# 2)selecting -15 to 15 lat, and maybe the three levels (125,100,85)
# 2a) alternativly, find the 3 lowest temperature for each lat/lon in 
# the level dimension
# 3) run a quadratic fit to the three points coldest points and select
# the minimum, put it in the lat/lon grid.
# 4) resample to monthly means and average over lat/lon and voila!
# T_all = xr.concat(T_list, 'time')
print('saving T.nc to {}'.format('/data11/ziskin/ERA5/'))
comp = dict(zlib=True, complevel=9)  # best compression
encoding = {var: comp for var in T.data_vars}
with ProgressBar():
    T.to_netcdf('/data11/ziskin/ERA5/T.nc', 'w', encoding=encoding)
