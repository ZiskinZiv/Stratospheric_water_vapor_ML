#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 13:12:22 2019

@author: shlomi
"""
import numpy as np
import xarray as xr
swoosh=xr.open_dataset(work_path+'swoosh_latpress-2.5deg.nc')
# com=swoosh.combinedanomfillanomh2oq
com=swoosh.combinedanomh2oq
com=com.sel(level=slice(83,81),lat=slice(-20,20))
weights = np.cos(np.deg2rad(com['lat'].values))
com_latmean = (weights*com).sum('lat')/sum(weights)
