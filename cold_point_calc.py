#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 18 18:53:12 2019

@author: shlomi
"""

from pathlib import Path

#1)take the min of 'level' for 6hr temperature model level/pressure level data
#2)resample to MS and mean('time')
#3)simple lat/lon mean
#4)remove the long term monthly mean (without sub the std) 
def compare_all(work_path):
    """compare era5, merra2, swoosh cold point -15 to 15 lat to swoosh
    h2o anom 82 hPa"""
    import xarray as xr
    import pandas as pd
    from aux_functions_strat import deseason_xr
    # first load swoosh data:
    sw = xr.open_dataset(work_path /
                         'swoosh-v02.6-198401-201812/swoosh-v02.6-198401-201812-latpress-2.5deg-L31.nc', decode_times=False)
    time = pd.date_range('1984-01-01', freq='MS', periods=sw.time.size)
    sw['time'] = time
    # cold point tropopause:
    cpt = sw['cptropt']
    cpt = cpt.sel(lat=slice(-15, 15))
    cpt = cpt.mean('lat')
    anom_cpt = deseason_xr(cpt, how='mean')
    # water vapor anomalies:
    wv = sw['combinedanomh2oq']
    wv_82 = wv.sel(level=82, method='nearest')
    wv_82 = wv_82.sel(lat=slice(-15, 15))
    wv_82 = wv_82.mean('lat')
    wv_82['time'] = time
    # merra2 just minimum:
    merra2 = get_coldest_point_merra2(work_path / 'MERRA2/', just_minimum=True)
    anom_merra2 = deseason_xr(merra2, how='mean')
    # era5 just minimum:
    era5 = get_coldest_point_era5(
        work_path, filename='T.nc', just_minimum=True)
    anom_era5 = deseason_xr(era5.t, how='mean')
    # put everything into a dataset:
    ds = wv_82.to_dataset(name=wv_82.name)
    ds['sw_cpt'] = anom_cpt
    ds['merra2'] = anom_merra2
    ds['era5'] = anom_era5
    # 3 months rolling mean:
    ds['merra2_smooth'] = ds.merra2.rolling(time=3, center=True).mean()
    ds['era5_smooth'] = ds.era5.rolling(time=3, center=True).mean()
    return ds


def get_coldest_point_era5(t_path, filename='T_1.nc', savepath=None, just_minimum=False):
    """create coldest point index by using era5 4xdaily data"""
    import xarray as xr
    import numpy as np
    import pandas as pd

    def quad(y):
        # note: apply always moves core dimensions to the end
        [a, b, c] = np.polyfit([1, 2, 3], y, 2)
        min_t = c - b**2 / (4 * a)
        for i in range(y.shape[1]):
            if np.abs(a[i]) < 1e-7 or min_t[i] <= y[0,i] or min_t[i] >= y[2,i]:
                min_t[i] = y[1,i]

        return min_t

    # 1)open mfdataset the temperature data
    # 2)selecting -15 to 15 lat, and maybe the three levels (125,100,85)
    # 2a) alternativly, find the 3 lowest temperature for each lat/lon in
    # the level dimension
    # 3) run a quadratic fit to the three points coldest points and select
    # the minimum, put it in the lat/lon grid.
    # 4) resample to monthly means and average over lat/lon and voila!
    if just_minimum:
        print('opening big T file ~ 2.9GB compressed...')
        T = xr.open_dataset(t_path / filename)
        da_list = []
        years = np.arange(1980, 2019)
        for year in years:
            T_s = T.sel(time=str(year))
            print('running year {}:'.format(year))
            # T_s['T'] = T_s['T'].transpose('lev', 'time', 'lon', 'lat')
            T_s = T_s.min('level')
            da = T_s.resample(time='MS').mean('time')
            da = da.mean('lon')
            da = da.mean('lat')
            da_list.append(da)
    else:
        print('opening big T file ~ 2.9GB compressed...')
        T = xr.open_dataset(t_path / filename)
        # T = T.sel(level=slice(70, 125))
        da_list = []
        years = np.arange(1980, 2019)
        for year in years:
            T_s = T.sel(time=str(year))
            print('running year {}:'.format(year))
            # T_s['T'] = T_s['T'].transpose('lev', 'time', 'lon', 'lat')
            T_s['t'] = T_s.t.transpose('level', 'time', 'lon', 'lat')
            T_s['t'] = T_s['t'].sel(level=slice(70, 125))
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
            print('saving to dataarray...')
            da = xr.DataArray(NU_result, dims=['time', 'lon', 'lat'])
            da['time'] = T_s.time
            da['lon'] = T_s.lon
            da['lat'] = T_s.lat
            print('resampling and averaging:')
            da = da.resample(time='MS').mean('time')
            da = da.mean('lon')
            da = da.mean('lat')
            da_list.append(da)
    da = xr.concat(da_list, 'time')
    if savepath is not None:
        comp = dict(zlib=True, complevel=9)  # best compression
        encoding = {var: comp for var in da.to_dataset(name='cold_point').data_vars}
        print('saving cold_point_era5.nc to {}'.format(savepath))
        da.to_dataset(name='cold_point').to_netcdf(savepath / 'cold_point_era5.nc', 'w', encoding=encoding)
    print('Done!')
    return da


def get_coldest_point_merra2(t_path, savepath=None, just_minimum=False):
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
        if just_minimum:
            T_s['T'] = T_s['T'].min('lev')
            da = T_s['T'].resample(time='MS').mean('time')
            da = da.mean('lon')
            da = da.mean('lat')
            da_list.append(da)
        else:
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
            da = da.resample(time='MS').mean('time')
            da = da.mean('lon')
            da = da.mean('lat')
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