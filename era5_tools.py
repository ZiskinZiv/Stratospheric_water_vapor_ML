#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 12:06:16 2018
# need to add ERA5 regriding 4Xdaily - to monthly mean data
@author: shlomi
"""


#def transform_l137_ml_to_pressure(ds, l137_path):
#    """transform model levels from int numbers to hPa full pressure levels"""
#    l137 = read_L137_to_ds(l137_path)
#    levels = ds.level.values
#    pf = l137.sel(n=levels).pf.values
#    ds['level'] = pf
#    return ds
#def ps_ufunc(x, a, b, n):
#    import xarray as xr
#    out = xr.apply_ufunc(pressure_from_ab, x, a, b, n, join='outer', output_core_dims=['n', 'time', 'latitude', 'longitude'])
#    return out


def transform_model_levels_to_pressure(path, field_da, plevel=85.0, mm=True):
    """takes a field_da (like t, u) that uses era5 L137 model levels and
    interpolates it using pf(only monthly means) to desired plevel"""
    # TODO: do scipy.interpolate instead of np.interp (linear)
    # TODO: add multiple plevel support
    import xarray as xr
    from aux_functions_strat import dim_intersection
    import numpy as np
    if mm:
        pf = xr.open_dataset(path / 'era5_full_pressure_mm_1979-2018.nc')
    pf = pf.pf
    levels = dim_intersection([pf, field_da], dropna=False, dim='level')
    pf = pf.sel(level=levels)
    field_da = field_da.sel(level=levels)
    field_da = field_da.transpose('time', 'latitude', 'longitude', 'level')
    field = field_da.values
    da = np.zeros((pf.time.size, pf.latitude.size, pf.longitude.size, 1),
                  'float32')
    pf = pf.values
    for t in range(pf.shape[0]):
        print(t)
        for lat in range(pf.shape[1]):
            for lon in range(pf.shape[2]):
                pressures = pf[t, lat, lon, :]
                vals = field[t, lat, lon, :]
                da[t, lat, lon, 0] = np.interp(plevel, pressures, vals)
    da = xr.DataArray(da, dims=['time', 'lat', 'lon', 'level'])
    da['level'] = [plevel]
    da['time'] = field_da['time']
    da['level'].attrs['long_name'] = 'pressure_level'
    da['level'].attrs['units'] = 'hPa'
    da['lat'] = field_da['latitude'].values
    da['lat'].attrs = field_da['latitude'].attrs
    da['lon'] = field_da['longitude'].values
    da['lon'].attrs = field_da['longitude'].attrs
    da.name = field_da.name
    da.attrs = field_da.attrs
    da = da.sortby('lat')
    comp = dict(zlib=True, complevel=9)  # best compression
    encoding = {var: comp for var in da.to_dataset(name=da.name).data_vars}
    filename = 'era5_' + da.name + '_' + str(int(plevel)) + 'hPa.nc'
    da.to_netcdf(path / filename, encoding=encoding)
    return da


def create_model_levels_map_from_surface_pressure(work_path):
    import numpy as np
    import xarray as xr
    """takes ~24 mins for monthly mean with 1.25x1.25 degree for full 137
    levels,~=2.9GB file. Don't try with higher sample rate"""
    ds = read_L137_to_ds(work_path)
    a = ds.a.values
    b = ds.b.values
    n = ds.n.values
    sp = xr.open_dataset(work_path / 'era5_SP_mm_1979-2018.nc')
    sp = sp.sp / 100.0
    pf = np.zeros((sp.time.size, sp.latitude.size, sp.longitude.size, len(a)),
                  'float32')
    sp_np = sp.values
    for t in range(sp.time.size):
        print(t)
        for lat in range(sp.latitude.size):
            for lon in range(sp.longitude.size):
                ps = sp_np[t, lat, lon]
                pf[t, lat, lon, :] = pressure_from_ab(ps, a, b, n)
    pf_da = xr.DataArray(pf, dims=['time', 'latitude', 'longitude', 'level'])
    pf_da.attrs['units'] = 'hPa'
    pf_da.attrs['long_name'] = 'full_pressure_level'
    pf_ds = pf_da.to_dataset(name='pf')
    pf_ds['level'] = n
    pf_ds['level'].attrs['long_name'] = 'model_level_number'
    pf_ds['latitude'] = sp.latitude
    pf_ds['latitude'].attrs = sp.latitude.attrs
    pf_ds['longitude'] = sp.longitude
    pf_ds['longitude'].attrs = sp.longitude.attrs
    pf_ds['time'] = sp.time
    comp = dict(zlib=True, complevel=9)  # best compression
    encoding = {var: comp for var in pf_ds.data_vars}
    pf_ds.to_netcdf(work_path / 'era5_full_pressure_mm_1979-2018.nc',
                    encoding=encoding)
    return pf_ds


def pressure_from_ab(ps, a, b, n):
    import numpy as np
    import xarray as xr
    ph_shape = a.shape
    ph = np.empty(ph_shape)
    pf = np.empty(len(a) - 1)
    pf_final = np.empty(ph_shape)
    ph = a / 100.0 + b * ps
    for i in n[0:-1]:
        pf[i] = 0.5 * (ph[i] + ph[i + 1])
    pf_final[0] = np.nan
    pf_final[1:] = pf
    # pf_da = xr.DataArray(pf_final, dims='n')
    # pf_da['n'] = n
    # pf_da.attrs['units'] = 'hPa'
    # pf_da['n'].attrs['long_name'] = 'model_level_number'
    return pf_final  # pf_da


def read_L137_to_ds(path):
    import pandas as pd
    l137_df = pd.read_csv(path / 'L137_model_levels_1976_climate.txt',
                          header=None, delim_whitespace=True, na_values='-')
    l137_df.columns = ['n', 'a', 'b', 'ph', 'pf', 'Geopotential Altitude',
                       'Geometric Altitude', 'Temperature', 'Density']
    l137_df.set_index('n')
    l137_df.drop('n', axis=1, inplace=True)
    l137_df.index.name = 'n'
    ds = l137_df.to_xarray()
    ds.attrs['long_name'] = 'L137 model levels and 1976 ICAO standard atmosphere 1976'
    ds.attrs['surface_pressure'] = 1013.250
    ds.attrs['pressure_units'] = 'hPa'
    ds.attrs['half_pressure_formula'] = 'ph(k+1/2) = a(k+1/2) + ps*b(k+1/2)'
    ds.attrs['full_pressure_formula'] = 'pf(k) = 1/2*(ph(k-1/2) + ph(k+1/2))'
    ds['n'].attrs['long_name'] = 'model_level_number'
    ds['a'].attrs['long_name'] = 'a_coefficient'
    ds['a'].attrs['units'] = 'Pa'
    ds['b'].attrs['long_name'] = 'b_coefficient'
    ds['ph'].attrs['long_name'] = 'half_pressure_level'
    ds['ph'].attrs['units'] = 'hPa'
    ds['pf'].attrs['long_name'] = 'full_pressure_level'
    ds['pf'].attrs['units'] = 'hPa'
    ds['Geopotential Altitude'].attrs['units'] = 'm'
    ds['Geometric Altitude'].attrs['units'] = 'm'
    ds['Temperature'].attrs['units'] = 'K'
    ds['Density'].attrs['units'] = 'kg/m^3'
    return ds


def compare_MERRA_ERA5(path='local', plot=False):
    import matplotlib.pyplot as plt
    from time_inds_for_MLR import get_BDC, get_T500
    import xarray as xr
    import aux_functions_strat as aux
    import os
    plt.close('all')
    ERA5_T = get_era5_fields(path, field='T_500', index=True)
    ERA5_T = ERA5_T.t_regrided_index
    ERA5_T.name = 'ERA5_T500'
    ERA5_BDC = get_era5_fields(path, field='BDC_54', index=True) # level 54 is closer to 70hpa than level 53
    ERA5_BDC = ERA5_BDC.mttpm_regrided_index
    ERA5_BDC.name = 'ERA5_BDC'
    ERA5_BDC = ERA5_BDC.to_dataset()
    ERA5_T = ERA5_T.to_dataset()
    ERA5_T['era5_anom_T'] = aux.deseason_xr(ERA5_T['ERA5_T500'])
    ERA5_BDC['era5_anom_BDC'] = aux.deseason_xr(ERA5_BDC['ERA5_BDC'])
    MERRA_T = get_T500(False)
    MERRA_T = MERRA_T.T20
    MERRA_T.name = 'MERRA_T500'
    MERRA_BDC = get_BDC(False)
    MERRA_BDC = MERRA_BDC.bdc20
    MERRA_BDC.name = 'MERRA_BDC'
    MERRA_BDC = MERRA_BDC.to_dataset()
    MERRA_T = MERRA_T.to_dataset()
    MERRA_T['merra_anom_T'] = aux.deseason_xr(MERRA_T['MERRA_T500'])
    MERRA_BDC['merra_anom_BDC'] = aux.deseason_xr(MERRA_BDC['MERRA_BDC'])
    t500_merged = xr.merge([MERRA_T, ERA5_T], join='inner')
    bdc_merged = xr.merge([MERRA_BDC, ERA5_BDC], join='inner')
    if plot:
        plot1 = t500_merged[['merra_anom_T', 'era5_anom_T']].to_dataframe().plot()
        plot2 = bdc_merged[['merra_anom_BDC', 'era5_anom_BDC']].to_dataframe().plot()
        plot3 = t500_merged[['MERRA_T500', 'ERA5_T500']].to_dataframe().plot()
        plot4 = bdc_merged[['MERRA_BDC', 'ERA5_BDC']].to_dataframe().plot()
    merged_all = xr.merge([ERA5_T, MERRA_T, MERRA_BDC, ERA5_BDC], join='outer')
    # do some stitching:
    merged_t = merged_all.era5_anom_T.combine_first(merged_all.merra_anom_T)
    # merged_t = merged_all.merra_anom_T.combine_first(merged_all.era5_anom_T)
    merged_t.name = 'ERA5_MERRA_T500'
    merged_bdc = merged_all.era5_anom_BDC.combine_first(merged_all.merra_anom_BDC)
    # merged_bdc = merged_all.merra_anom_BDC.combine_first(merged_all.era5_anom_BDC)
    merged_bdc.name = 'ERA5_MERAA_BDC'
    merged_all['merged_t'] = merged_t
    merged_all['merged_bdc'] = merged_bdc
    if plot:
        plot5 = merged_all[['merged_t', 'merra_anom_T', 'era5_anom_T']].to_dataframe().plot()
        plot6 = merged_all[['merged_bdc', 'merra_anom_BDC', 'era5_anom_BDC']].to_dataframe().plot()
        plt.show()
    path = os.getcwd() + '/'
    merged_t.to_netcdf(path + 'era5_merra_T500.nc', 'w')
    print('saved merra_t in local dir.')
    merged_bdc.to_netcdf(path + 'era5_merra_BDC.nc', 'w')
    print('saved merra_bdc in local dir.')
    return merged_all


def regrid_era5_back_to_center_lon_coords(da, lon_dim='lon', lat_dim='lat'):
    import xesmf as xe
    import xarray as xr
    from aux_functions_strat import copy_coords_attrs
    attrs = da.attrs
    da = da.transpose(..., lat_dim, lon_dim)
    lon = da[lon_dim].values - 180
    lat = da[lat_dim].values
    ds_out = xr.Dataset({'lat': (['lat'], lat), 'lon': (['lon'], lon)})
    regridder = xe.Regridder(da, ds_out, 'bilinear')
    print(regridder)
    dr_out = regridder(da)
    for key, value in attrs.items():
        dr_out.attrs[key] = value
    dr_out = copy_coords_attrs(da, dr_out, verbose=True)
    return dr_out


def regrid_era5(da):
    """regrid era5 dataarray to center coordinates."""
    import numpy as np
    import aux_functions_strat as aux
    import xarray as xr
    area = aux.grid_seperation_xr(1.25, 1.25, lon_start=0.0)  # i d/l from ecmwf with this resolution
    # The center coords from grid calculation:
    req_lat = area.lat_center.values
    req_lon = area.lon_center.values
    native_lat = da.lat.values
    native_lon = da.lon.values
    da_name = da.name
    attrs = da.attrs
    da_rgr = da.copy()
    da_rgr = da_rgr.to_dataset()
    da_rgr = da_rgr.rename({'lat': 'lat_outer'})
    da_rgr = da_rgr.rename({'lon': 'lon_outer'})
    data = da.values
    if 'level' in da.dims:
        data_out = np.empty((data.shape[0], data.shape[1], len(req_lat), len(req_lon)))
        for time in range(data.shape[0]):
            print(time)
            for level in range(data.shape[1]):
                data_out[time, level, :, :] = aux.generic_regrid(data[time, level, :, :],
                                                                 native_lat,
                                                                 native_lon,
                                                                 req_lat,
                                                                 req_lon, 3)
                da_rgr[da_name + '_regrided'] = xr.DataArray(data_out,
                                                             coords=[da.time,
                                                                     da.level,
                                                                     req_lat,
                                                                     req_lon],
                                                             dims=['time',
                                                                   'level',
                                                                   'lat',
                                                                   'lon'])
    else:
        data_out = np.empty((data.shape[0], len(req_lat), len(req_lon)))
        for time in range(data.shape[0]):
            print(time)
            data_out[time, :, :] = aux.generic_regrid(data[time, :, :], native_lat,
                                                      native_lon, req_lat, req_lon, 3)
            da_rgr[da_name + '_regrided'] = xr.DataArray(data_out,
                                                         coords=[da.time,
                                                                 req_lat,
                                                                 req_lon],
                                                         dims=['time', 'lat',
                                                                       'lon'])
    da_rgr[da_name + '_regrided'].attrs = attrs
    da_rgr['lat'].attrs = da_rgr['lat_outer'].attrs
    da_rgr['lon'].attrs = da_rgr['lon_outer'].attrs
    return da_rgr


def get_era5_fields(path='local', field='T_all', index=True):
    import xarray as xr
    import os
    import glob
    from aux_functions_strat import text_green
    import aux_functions_strat as aux
    if path == 'local':
        path = os.getcwd() + '/'
    xr_list = []
    for filename in glob.iglob(path + 'era5_moda_' + field + '*.nc'):
        xr_list.append(xr.open_dataarray(filename))
        text_green('Proccessing file:' + filename)
    xarray = xr.concat(xr_list, dim='time')
    xarray = aux.xr_rename_sort(xarray, lon_roll=False)
    xarray = aux.xr_order(xarray)
    xarray = regrid_era5(xarray)
    rgr_name = [x for x in xarray.data_vars.keys() if 'regrided' in x][0]
    da = xarray[rgr_name]
    da = da.reset_coords(drop=True)
    da = aux.xr_rename_sort(da, lon_roll=True)
    da.name = field.split('_')[0]
    if index:
        xarray[rgr_name + '_index'] = aux.xr_weighted_mean(xarray[rgr_name].sel(lat=slice(-20, 20)))
    comp = dict(zlib=True, complevel=9)  # best compression
    encoding = {var: comp for var in da.to_dataset().data_vars}
    da.to_netcdf(path + 'ERA5_' + field + '.nc', 'w', encoding=encoding)
    # xarray[rgr_name + '_index'] = aux.area_weighted_xr(xarray[rgr_name].sel(lat=slice(-20, 20)))
    return da


def get_hourly_era5_fields(path, field='U', mean=True):
    import xarray as xr
    import aux_functions_strat as aux
    xarray = xr.open_mfdataset(path + 'era5_' + field + '_*.nc')
    name = [x for x in xarray.data_vars.keys()][0]
    xarray = xarray.to_array(name=name).squeeze(drop=True)
    xarray = aux.xr_rename_sort(xarray, lon_roll=False)
    xarray = aux.xr_order(xarray)
    xarray = xarray.resample(time='D').mean('time')
    xarray = xarray.resample(time='MS').mean('time')
    xarray = regrid_era5(xarray)
    rgr_name = [x for x in xarray.data_vars.keys() if 'regrided' in x][0]
    da = xarray[rgr_name]
    da = da.reset_coords(drop=True)
    da = aux.xr_rename_sort(da, lon_roll=True)
    da.name = field.split('_')[0]
    if mean:
        xarray[rgr_name + '_mean'] = aux.xr_weighted_mean(xarray[rgr_name].sel(lat=slice(-5, 5)))
    comp = dict(zlib=True, complevel=9)  # best compression
    encoding = {var: comp for var in da.to_dataset().data_vars}
    da.to_netcdf(path + 'ERA5_' + field + '.nc', 'w', encoding=encoding)
    # xarray[rgr_name + '_index'] = aux.area_weighted_xr(xarray[rgr_name].sel(lat=slice(-20, 20)))
    return da


def proccess_era5_fields(path, pre_names, post_name, mean=True, savepath=None):
    import xarray as xr
    import aux_functions_strat as aux
    import xesmf as xe
    from pathlib import Path
    if savepath is None:
        savepath = Path().cwd()
    xarray = xr.open_mfdataset(str(path) + 'era5_' + pre_names + '_*.nc')
    name = [x for x in xarray.data_vars.keys()][0]
    xarray = xarray.to_array(name=name).squeeze(drop=True)
    xarray = aux.xr_rename_sort(xarray, lon_roll=False)
    xarray = aux.xr_order(xarray)
    area = aux.grid_seperation_xr(1.25, 1.25, lon_start=0.0)
    ds_out = xr.Dataset({'lat': (['lat'], area.lat_center.values),
                         'lon': (['lon'], area.lon_center.values), })
    regridder = xe.Regridder(xarray.to_dataset(name=name), ds_out, 'bilinear')
    da = regridder(xarray)
    regridder.clean_weight_file()
    # xarray = regrid_era5(xarray)
    # rgr_name = [x for x in xarray.data_vars.keys() if 'regrided' in x][0]
    # da = xarray[rgr_name]
    da = da.reset_coords(drop=True)
    da = aux.xr_rename_sort(da, lon_roll=True)
    da.name = pre_names.split('_')[0]
    if mean:
        da = aux.xr_weighted_mean(da.sel(lat=slice(-5, 5)))
    comp = dict(zlib=True, complevel=9)  # best compression
    encoding = {var: comp for var in da.to_dataset().data_vars}
    da.to_netcdf(savepath + 'era5_' + post_name + '.nc', 'w',
                 encoding=encoding)
    print('saved ' + 'era5_' + post_name + '.nc' + ' to path:' + str(savepath))
    return da
