#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 22 11:21:00 2019

@author: shlomi
"""


def concat_to_years():
    from pathlib import Path
    import numpy as np
    import xarray as xr
    import pandas as pd
    data11 = Path('/data11/ziskin/MERRA2/')
    years = np.arange(2001, 2019)
    for year in years:
        ds_year = []
        print('proccessing year {}:'.format(year))
        part = '100'
        if year > 1991:
            part = '200'
        if year > 2000:
            part = '300'
        if year > 2010:
            part = '400'
        year_date = pd.date_range(
            str(year) + '-01-01',
            str(year) + '-12-31',
            freq='D')
        for date in year_date:
            filename = 'MERRA2_' + part + '.inst6_3d_ana_Nv.' + \
                date.strftime('%Y%m%d') + '.SUB.nc'
            try:
                merra2 = xr.open_dataset(data11.as_posix() + '/' + filename)
            except FileNotFoundError:
                print('{} not found...skipping'.format(filename))
                continue
            ds_year.append(merra2)
        ds = xr.concat(ds_year, 'time')
        comp = dict(zlib=True, complevel=9)  # best compression
        encoding = {var: comp for var in ds.data_vars}
        file_to_save = 'MERRA2_T_cold_' + str(year) + '.nc'
        ds.to_netcdf(data11 / file_to_save, 'w', encoding=encoding)
    return


def concat_to_one_big_file():
    from pathlib import Path
    import xarray as xr
    data11 = Path('/data11/ziskin/MERRA2/')
    from dask.diagnostics import ProgressBar
    merra2 = xr.open_mfdataset(data11.as_posix() + '/' 'MERRA2_T_cold_*.nc')
    comp = dict(zlib=True, complevel=9)  # best compression
    encoding = {var: comp for var in merra2.data_vars}
    file_to_save = 'MERRA2_Tcold_BIG.nc'
    with ProgressBar():
        merra2.to_netcdf(data11 / file_to_save, 'w', encoding=encoding)
    return

concat_to_one_big_file()
