#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 18 18:53:12 2019

@author: shlomi
"""

from pathlib import Path


def get_coldest_point_era5(t_path, filename='T_1.nc', savepath=None):
    """create coldest point index by using era5 4xdaily data"""
    import xarray as xr
    import numpy as np
    import pandas as pd

    def quad(y):
        # note: apply always moves core dimensions to the end
        [a, b, c] = np.polyfit([70, 100, 125], y, 2)
        min_t = c - b**2 / (4 * a)
        min_t = np.array([y[1, i] for i in np.arange(
            y.shape[1]) if min_t[i] <= y[0, i] or min_t[i] >= y[2, i]])
        return min_t

    # 1)open mfdataset the temperature data
    # 2)selecting -15 to 15 lat, and maybe the three levels (125,100,85)
    # 2a) alternativly, find the 3 lowest temperature for each lat/lon in
    # the level dimension
    # 3) run a quadratic fit to the three points coldest points and select
    # the minimum, put it in the lat/lon grid.
    # 4) resample to monthly means and average over lat/lon and voila!
    print('opening big T file ~ 2.9GB compressed...')
    T = xr.open_dataset(t_path / filename)
    # T = T.sel(level=slice(70, 125))
    years = np.arange(1979, 2018 + 5, 4)
    years = [str(x) for x in years]
    T_list = []
    time_list = []
    for i in np.arange(len(years) - 1):
        time_list.append(T.time.sel(time=slice(
            years[i], str(int(years[i + 1]) - 1))))
        T_list.append(T.sel(time=slice(years[i], str(int(years[i + 1]) - 1))))
    da_list = []
    for T_s in T_list:
        min_year = pd.to_datetime(T_s.time.values.min()).year
        max_year = pd.to_datetime(T_s.time.values.max()).year
        print('running portion of T, years {} to {}'.format(min_year, max_year))
        T_s['t'] = T_s.t.transpose('level', 'time', 'lon', 'lat')
        print('getting numpy array...')
        NU = T_s.t.values
        print('reshaping...')
        NU_reshaped = NU.reshape(NU.shape[0], NU.shape[1] * NU.shape[2] *
                                 NU.shape[3])
        print('allocating new array...')
        NU_result = np.empty((NU_reshaped.shape[1]))
        print('running quad...')
        NU_result = quad(NU_reshaped)
        print('reshaping back...')
        NU_result = NU_result.reshape(NU.shape[1], NU.shape[2], NU.shape[3])
    #    # main loop:
    #    for time, lon, lat in product(range(NU.shape[0]), range(NU.shape[1]),
    #                                  range(NU.shape[2])):
    #        NU_res[time, lon, lat] = quad(NU[time, lon, lat, :])
        # put data in dataarray:
        print('saving to dataarray...')
        da = xr.DataArray(NU_result, dims=['time', 'lon', 'lat'])
        da['time'] = T_s.time
        da['lon'] = T_s.lon
        da['lat'] = T_s.lat
        da_list.append(da)
    da = xr.concat(da_list, 'time')
    if savepath is not None:
        comp = dict(zlib=True, complevel=9)  # best compression
        encoding = {var: comp for var in da.to_dataset(name='cold_point').data_vars}
        print('saving cold_point_era5.nc to {}'.format(savepath))
        da.to_dataset(name='cold_point').to_netcdf(savepath / 'cold_point_era5.nc', 'w', encoding=encoding)
    print('Done!')
    return da


def get_coldest_point_merra2(t_path, savepath=None):
    """create coldest point index by using MERRA2 4xdaily data"""
    import xarray as xr
    import numpy as np
    import pandas as pd

    def quad(y):
        # note: apply always moves core dimensions to the end
        [a, b, c] = np.polyfit([38, 39, 40], y, 2)
        min_t = c - b**2 / (4 * a)
        for i in range(y.shape[1]):
            if np.abs(a[i]) < 1e-7 or min_t[i] <= y[0,i] or min_t[i] >= y[2,i]:
                min_t[i] = y[1,i]

        # print(y.shape)
        # min_t = np.array([y[1, i] for i in np.arange(
        #         y.shape[1]) if min_t[i] <= y[0, i] or min_t[i] >= y[2, i] or np.isinf(min_t[i])])
        return min_t
    da_list = []
    years = np.arange(1980, 2019)
    for year in years:
        filename = 'MERRA2_T_cold_' + str(year) + '.nc'
        print(filename)
        T_s = xr.open_dataset(t_path / filename)
        print('running year {}:'.format(year))
        # T_s['T'] = T_s['T'].transpose('lev', 'time', 'lon', 'lat')
        print('getting numpy array...')
        NU = T_s['T'].values
        NU = np.transpose(NU, axes=(1, 0, 3, 2))
        NU = NU[1:4, :, :, :]
        print('reshaping...')
        NU_reshaped = NU.reshape(NU.shape[0], NU.shape[1] * NU.shape[2]
                                 * NU.shape[3])
        print('allocating new array...')
        NU_result = np.empty((NU_reshaped.shape[1]))
        print('running quad...')
        NU_result = quad(NU_reshaped)
        print('reshaping back...')
        NU_result = NU_result.reshape(NU.shape[1], NU.shape[2], NU.shape[3])
    #    # main loop:
    #    for time, lon, lat in product(range(NU.shape[0]), range(NU.shape[1]),
    #                                  range(NU.shape[2])):
    #        NU_res[time, lon, lat] = quad(NU[time, lon, lat, :])
        # put data in dataarray:
        print('saving to dataarray...')
        da = xr.DataArray(NU_result, dims=['time', 'lon', 'lat'])
        da['time'] = T_s.time
        da['lon'] = T_s.lon
        da['lat'] = T_s.lat
        da_list.append(da)
    da = xr.concat(da_list, 'time')
    if savepath is not None:
        comp = dict(zlib=True, complevel=9)  # best compression
        encoding = {
            var: comp for var in da.to_dataset(
                name='cold_point').data_vars}
        print('saving cold_point_merra2.nc to {}'.format(savepath))
        da.to_dataset(name='cold_point').to_netcdf(savepath
                                                   / 'cold_point_merra2.nc',
                                                   'w', encoding=encoding)
    print('Done!')
    return da
# path = Path('/data11/ziskin/MERRA2/')
# da = get_coldest_point_merra2(path, path)