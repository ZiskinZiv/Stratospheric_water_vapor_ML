#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 13:12:22 2019

@author: shlomi
"""
import numpy as np
import xarray as xr
import aux_functions_strat as aux
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
swoosh = xr.open_dataset(work_path + 'swoosh_latpress-2.5deg.nc')
haloe = xr.open_dataset(work_path +
                        'swoosh-v02.6-198401-201812/swoosh-v02.6-198401-201812-latpress-2.5deg-L31.nc', decode_times=False)
haloe['time'] = swoosh.time
haloe_names = [x for x in haloe.data_vars.keys()
               if 'haloe' in x and 'h2o' in x]
haloe = haloe[haloe_names].sel(level=slice(83, 81), lat=slice(-20, 20))
# com=swoosh.combinedanomfillanomh2oq
com = swoosh.combinedanomfillanomh2oq
com = com.sel(level=slice(83, 81), lat=slice(-20, 20))
weights = np.cos(np.deg2rad(com['lat'].values))
com_latmean = (weights * com).sum('lat') / sum(weights)
com_latmean_2M_lagged = com_latmean.shift(time=-2)
com_latmean_2M_lagged.name = com_latmean.name + ' + 2M lag'
haloe_latmean = (weights * haloe.haloeanomh2oq).sum('lat') / sum(weights)
era40 = xr.open_dataarray(work_path + 'ERA40_T_mm_eq.nc')
era40 = era40.sel(level=100)
weights = np.cos(np.deg2rad(era40['lat'].values))
era40_latmean = (weights * era40).sum('lat') / sum(weights)
era40anom_latmean = aux.deseason_xr(era40_latmean)
era40anom_latmean.name = 'era40_100hpa_anomalies'
era5 = xr.open_dataarray(work_path + 'ERA5_T_eq_all.nc')
cold_point = era5.sel(level=slice(150, 50)).min(['level', 'lat',
                                                 'lon'])
cold_point = aux.deseason_xr(cold_point)
cold_point.name = 'cold_point_from_era5'
era5 = era5.mean('lon').sel(level=100)
weights = np.cos(np.deg2rad(era5['lat'].values))
era5_latmean = (weights * era5).sum('lat') / sum(weights)
era5anom_latmean = aux.deseason_xr(era5_latmean)
era5anom_latmean.name = 'era5_100hpa_anomalies'
merra = xr.open_dataarray(work_path + 'T_regrided.nc')
merra['time'] = pd.date_range(start='1979', periods=merra.time.size, freq='MS')
merra = merra.mean('lon').sel(lat=slice(-20, 20), level=100)
weights = np.cos(np.deg2rad(merra['lat'].values))
merra_latmean = (weights * merra).sum('lat') / sum(weights)
merra_latmean.name = 'merra'
merraanom_latmean = aux.deseason_xr(merra_latmean)
merraanom_latmean.name = 'merra_100hpa_anomalies'
to_compare = xr.merge([com_latmean.squeeze(drop=True),
                       com_latmean_2M_lagged.squeeze(drop=True),
                       cold_point,
                       era40anom_latmean.squeeze(drop=True),
                       era5anom_latmean.squeeze(drop=True),
                       merraanom_latmean.squeeze(drop=True)])
to_compare.to_dataframe().plot()
plt.figure()
sns.heatmap(to_compare.to_dataframe().corr(), annot=True)
plt.subplots_adjust(left=0.35, bottom=0.4, right=0.95)
to_compare.sel(time=slice('1993', '2003')).to_dataframe().plot()
plt.figure()
sns.heatmap(to_compare.sel(time=slice('1993', '2003')).to_dataframe().corr(),
            annot=True)
plt.subplots_adjust(left=0.35, bottom=0.4, right=0.95)
