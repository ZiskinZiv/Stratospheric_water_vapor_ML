#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 11:23:16 2019

@author: ziskin
"""

from strat_paths import work_chaim
merra2_path = work_chaim/'MERRA2'


def produce_india_pollution_index(path, data_type='CO', plot=True):
    import xarray as xr
    from aux_functions_strat import lat_mean
    import matplotlib.pyplot as plt
    if data_type == 'CO':
        full_ds = xr.load_dataset(path / 'MERRA2_carbon.nc')
        full_ds = full_ds.sortby('time')
        das = ['COCL', 'COEM', 'COLS', 'COPD', 'COSC', 'TO3']
        title = 'Global vs. India CO indicies and total ozone'
    elif data_type == 'BC':
        full_ds = xr.load_dataset(path / 'MERRA2_aerosol.nc')
        full_ds = full_ds.sortby('time')
        das = ['BCANGSTR', 'BCCMASS', 'BCEXTTAU', 'BCFLUXU', 'BCFLUXV',
               'BCSCATAU', 'BCSMASS']
        title = 'Global vs. India BC indicies'
    elif data_type == 'OC':
        full_ds = xr.load_dataset(path / 'MERRA2_aerosol.nc')
        full_ds = full_ds.sortby('time')
        das = ['OCANGSTR', 'OCCMASS', 'OCEXTTAU', 'OCFLUXU', 'OCFLUXV',
               'OCSCATAU', 'OCSMASS']
        title = 'Global vs. India BC indicies'
    elif data_type == 'DUST':
        full_ds = xr.load_dataset(path / 'MERRA2_aerosol.nc')
        full_ds = full_ds.sortby('time')
        das = ['DUANGSTR', 'DUCMASS', 'DUCMASS25', 'DUEXTT25', 'DUEXTTAU',
               'DUFLUXU', 'DUFLUXV', 'DUSCAT25', 'DUSCATAU', 'DUSMASS',
               'DUSMASS25']
        title = 'Global vs. India dust indicies'
    elif data_type == 'AER':
        full_ds = xr.load_dataset(path / 'MERRA2_aerosol.nc')
        full_ds = full_ds.sortby('time')
        das = ['TOTANGSTR', 'TOTEXTTAU', 'TOTSCATAU']
        title = 'Global vs. India total aerosol indicies'
    dss = full_ds[das]
    # get units:
    units = [x.attrs['units'] for x in dss.data_vars.values()]
    long_names = [x.attrs['long_name'] for x in dss.data_vars.values()]
    # first get a -20 to 20 band lat mean and global lon mean index:
    global_dss = lat_mean(dss.mean('lon', keep_attrs=True))
    names = [x for x in global_dss]
    for name, long_name, unit in zip(names, long_names, units):
        print('{}: {} in {}'.format(name, long_name, unit))
    # rename vars:
    g_names = ['global_{}'.format(x) for x in names]
    nnd = dict(zip(names, g_names))
    global_dss = global_dss.rename(nnd)
    # second get india's geo-mean:
    india_dss = dss.sel(lat=slice(5, 30), lon=slice(65, 95))
    india_dss = lat_mean(india_dss.mean('lon', keep_attrs=True))
    # rename vars:
    i_names = ['india_{}'.format(x) for x in names]
    nnd = dict(zip(names, i_names))
    india_dss = india_dss.rename(nnd)
    # merge all:
    ds = xr.merge([india_dss, global_dss])
    if plot:
        fig, axes = plt.subplots(nrows=len(
            i_names), ncols=1, sharex=True, figsize=(20, 16))
        for i, var_name in enumerate(list(zip(g_names, i_names))):
            var_name = [x for x in var_name]
            ds[var_name].to_dataframe().plot(
                ax=axes[i]).legend(
                loc='upper left', framealpha=0.5)
            axes[i].set_ylabel(units[i])
        fig.suptitle(title, fontweight='bold')
        fig.tight_layout()
        fig.subplots_adjust(top=0.95)
    return ds
