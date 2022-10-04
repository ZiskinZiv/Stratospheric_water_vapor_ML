#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 13:06:36 2018

@author: shlomi
OK
1)fix and get time series of SAOD volcanic index - see how kuchar did it
he did it SAD = surface area density of aerosols at 54 hPa
2)use enso3.4 anom for enso
3)use singapore qbo(level=50) for qbo
4)use solar f10.7 for solar
5) for MLR use the MERRA grid (reduced-interpolated), and the swoosh different grids
6) prepare MLR framework and run on cluster
more analysis:
    1) include BDC from papar (reconstruct 70hpa - file from chaim weighted mean for tropics after zonal mean i.e time_series only)
    2) include Temp (500 hpa) from merra time_series mean -add feature to choose the "tropical" definition where the weighted mean should be taken i.e. -20 to 20 lat to -5 to 5
    1) and 2) can replace enso
    3) regardless print a heatmap of regressors corr.
    4) mask or remove pinatubo in volcalnic series or in all times
        1) do this menually this time...
    5) adding a time series reconstruction to results
        1)already exists...apply on demand bc its a lot of data
        2) use plotter subroutines to reconstruct the time-series fields.
        3) even bettter : work with psyplot gui to select and plot what i want
    6) change the data retrievel scheme to include different regressos:
        1) first save the regressors in file format - regressors#1 and have a file;
        2) MLR_all will run on all regressors list and append _R# suffix before .nc;
        need to change save and load routines to accomplish this.
new methodology:
    1) write single function to produce or load single or datasets of
    regressors using _produce or _load, remember to save with _index.nc suffix
    2) use load_all_regressors to load all of the regressors in reg_path
    3) select specific regressors and do anomaly(after time slice)
"""

from strat_paths import work_chaim
from strat_paths import cwd
reg_path = cwd / 'regressors'


def print_saved_file(name, path):
    print('{} was saved to {}'.format(name, path))
    return


def load_all_regressors(loadpath=reg_path):
    """load all regressors(end with _index.nc') from loadpath to dataset"""
    import xarray as xr
    from collections import OrderedDict
    from aux_functions_strat import path_glob
    da_list = []
    da_list_from_ds = []
    files = sorted(path_glob(reg_path, '*index.nc'))
    for file in files:
        name = file.as_posix().split(
            '/')[-1].split('.')[0].replace('_index', '')
        try:
            da = xr.load_dataarray(file)
            da = da.reset_coords(drop=True)
            da.name = name
            da_list.append(da)
        except ValueError:
            ds = xr.load_dataset(file)
            for da in ds.data_vars.values():
                da = da.reset_coords(drop=True)
                try:
                    da.name = da.attrs['name']
                except KeyError:
                    da.name = name + '_' + da.name
                    # avoid name repetition:
                    da.name = "_".join(OrderedDict.fromkeys(da.name.split('_')))
                da_list_from_ds.append(da)
    for das in da_list_from_ds:
        da_list.append(das)
    ds = xr.merge(da_list)
    return ds


def prepare_regressors(name='Regressors', plot=True, save=False,
                       rewrite_file=True, normalize=False, savepath=None,
                       rolling=None):
    """get all the regressors and prepare them save to file.
    replaced prepare_regressors for MLR function"""
    import aux_functions_strat as aux
    import xarray as xr
    import seaborn as sns
    import matplotlib.pyplot as plt
    import os
    from pathlib import Path
    # regressors names and filenames dict:
    reg_file_dict = {'bdc': 'era5_bdc_index.nc',
                     't500': 'era5_t500_index.nc',
                     'enso': 'anom_nino3p4_index.nc',
                     'solar': 'solar_10p7cm_index.nc',
                     'vol': 'vol_index.nc',
                     'qbo': 'era5_qbo_index.nc',
                     'olr': 'olr_index.nc',
                     'ch4': 'ch4_index.nc',
                     'wind': 'era5_wind_shear_index.nc',
                     'cold': 'cpt_index.nc',
                     'aod': 'merra2_totexttau_index.nc'}
    if savepath is None:
        savepath = Path().cwd() / 'regressors/'
    # aod:
    aod = load_regressor(reg_file_dict['aod'], plot=False, dseason=False)
    aod.name = 'aod'
    # bdc:
    bdc = load_regressor(reg_file_dict['bdc'], plot=False, deseason=True)
    if rolling is not None:
        bdc = bdc.rolling(time=3).mean()
    bdc.name = 'bdc'
    # t500
    t500 = load_regressor(reg_file_dict['t500'], plot=False, deseason=True)
    if rolling is not None:
        t500 = t500.rolling(time=3).mean()
    t500.name = 't500'
    # ENSO
    enso = load_regressor(reg_file_dict['enso'], plot=False, deseason=False)
    enso.name = 'enso'
    # SOLAR
    solar = load_regressor(reg_file_dict['solar'], plot=False, deseason=False)
    solar.name = 'solar'
    # Volcanic forcing
    vol = load_regressor(reg_file_dict['vol'], plot=False, deseason=False)
    vol.name = 'vol'
    # get the qbo 2 pcs:
    qbo = load_regressor(reg_file_dict['qbo'], plot=False, deseason=False,
                         is_dataset=True)
    qbo_1 = qbo['qbo_1']
    qbo_2 = qbo['qbo_2']
    # get GHG:
    # ghg = load_regressor(reg_file_dict['ghg'], plot=False, deseason=False)
    # ghg.name = 'ghg'
    # get cold point:
    cold = load_regressor(reg_file_dict['cold'], plot=False, deseason=True)
    if rolling is not None:
        cold = cold.rolling(time=3).mean()
    cold.name = 'cold'
    # get olr:
    olr = load_regressor(reg_file_dict['olr'], plot=False, deseason=True)
    olr.name = 'olr'
    # get ch4:
    ch4 = load_regressor(reg_file_dict['ch4'], plot=False, deseason=False,
                         normalize=True)
    ch4.name = 'ch4'
    # get wind_shear:
    wind = load_regressor(reg_file_dict['wind'], plot=False, deseason=False)
    wind.name = 'wind'
    da_list = [x for x in reg_file_dict.keys() if x != 'qbo']
    da_list += ['qbo_1', 'qbo_2']
    ds = xr.Dataset()
    for da_name in da_list:
        ds[da_name] = locals()[da_name]
    # fix vol and ch4
    ds['vol'] = ds['vol'].fillna(1.31)
    ds = ds.reset_coords(drop=True)
    # ds['ch4'] = ds['ch4'].fillna(0.019076 + 1.91089)
#    if poly is not None:
#        da = ds.to_array(dim='regressors').dropna(dim='time').T
#        da = poly_features(da, feature_dim='regressors', degree=poly,
#                           interaction_only=False, include_bias=False,
#                           normalize_poly=False)
#        ds = da.to_dataset(dim='regressors')
#        name = 'Regressors_d' + str(poly)
#    else:
#     name = 'Regressors'
    if normalize:
        ds = ds.apply(aux.normalize_xr, norm=1,
                      keep_attrs=True, verbose=False)
    if save:
        if rewrite_file:
            try:
                os.remove(str(savepath) + name + '.nc')
            except OSError as e:  # if failed, report it back to the user
                print("Error: %s - %s." % (e.filename, e.strerror))
            print('Updating ' + name + '.nc' + ' in ' + str(savepath))
        filename = name + '.nc'
        ds.to_netcdf(savepath / filename)
        print_saved_file(name, savepath)
    if plot:
        le = len(ds.data_vars)
        df = ds.to_dataframe()
        df.plot()
        plt.figure()
        if le <= 20:
            sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='bwr',
                        center=0.0)
        else:
            sns.heatmap(df.corr(), cmap='bwr', center=0.0)
    return ds


def load_regressor(regressor_file, plot=True, deseason=True, normalize=False,
                   path=None, is_dataset=False):
    """loads a regressor from regressors folder. you can deseason it,
    plot it, normalize it, etc..."""
    import xarray as xr
    from pathlib import Path
    if path is None:
        path = Path().cwd() / 'regressors/'
    if is_dataset:
        reg = xr.open_dataset(path / regressor_file)
    else:
        reg = xr.open_dataarray(path / regressor_file)
    if deseason:
        from aux_functions_strat import deseason_xr
        reg = deseason_xr(reg, how='mean')
    if normalize:
        from aux_functions_strat import normalize_xr
        # normalize = remove mean and divide by std
        reg = normalize_xr(reg, verbose=False)
    if plot:
        if is_dataset:
            reg.to_pandas().plot()
        else:
            reg.plot()
    return reg


def split_anom_nino3p4_to_EN_LN_neutral(loadpath=reg_path, savepath=None):
    ds = load_all_regressors(loadpath)
    enso = ds['anom_nino3p4'].dropna('time')
    EN = enso[enso >= 0.5].reindex(time=enso['time']).fillna(0)
    EN.attrs['action'] = 'only EN (ENSO >=0.5) kept, other is 0.'
    LN = enso[enso <= -0.5].reindex(time=enso['time']).fillna(0)
    LN.attrs['action'] = 'only LN (ENSO <=-0.5) kept, other is 0.'
    neutral = enso[(enso > -0.5) & (enso < 0.5)
                   ].reindex(time=enso['time']).fillna(0)
    neutral.attrs['action'] = 'only neutENSO (ENSO<0.5 & ENSO>-0.5) kept, other is 0.'
    if savepath is not None:
        EN.to_netcdf(savepath / 'EN_index.nc')
        LN.to_netcdf(savepath / 'LN_index.nc')
        neutral.to_netcdf(savepath / 'neutENSO_index.nc')
    return EN, LN, neutral


def _produce_wind_shear(source='singapore', savepath=None):
    import xarray as xr
    from pathlib import Path
    if source == 'singapore':
        u = _download_singapore_qbo(path=savepath)
        filename = 'singapore_wind_shear_index.nc'
    elif source == 'era5':
        u = xr.open_dataarray(savepath / 'ERA5_U_eq_mean.nc')
        filename = 'era5_wind_shear_index.nc'
    wind_shear = u.diff('level').sel(level=70)
    wind_shear.name = 'wind_shear'
    if savepath is not None:
        wind_shear.to_netcdf(savepath / filename)
        print_saved_file(filename, savepath)
    return wind_shear


def _download_CH4(filename='ch4_mm.nc', loadpath=None,
                  trend=False, savepath=None, interpolate=False):
    import xarray as xr
    import pandas as pd
    import wget
    filepath = loadpath / filename
    if filepath.is_file():
        print('CH4 monthly means from NOAA ERSL already d/l and saved!')
        # read it to data array (xarray)
        ch4_xr = xr.open_dataset(loadpath / filename)
        # else d/l the file and fObsirst read it to df (pandas),
        # then to xarray then save as nc:
    else:
        link = 'ftp://aftp.cmdl.noaa.gov/products/trends/ch4/ch4_mm_gl.txt'
        wget.download(link, out=loadpath.as_posix() + '/ch4_mm_gl.txt')
        ch4_df = pd.read_csv(loadpath / 'ch4_mm_gl.txt', delim_whitespace=True,
                             comment='#',
                             names=['year', 'month', 'decimal', 'average',
                                    'average_unc', 'trend', 'trend_unc'])
        print('Downloading CH4 monthly means from NOAA ERSL website...')
        ch4_df = ch4_df.drop(0)
        idx = pd.to_datetime(dict(year=ch4_df.year, month=ch4_df.month,
                                  day='01'))
        ch4_df = ch4_df.set_index(idx)
        ch4_df = ch4_df.drop(ch4_df.iloc[:, 0:3], axis=1)
        ch4_df = ch4_df.rename_axis('time')
        ch4_xr = xr.Dataset(ch4_df)
        ch4_xr.attrs['long_name'] = 'Monthly averages of CH4 concentrations'
        ch4_xr.attrs['units'] = 'ppb'
#    if savepath is not None:
#        ch4_xr.to_netcdf(savepath / filename)
#        print('Downloaded CH4 monthly means data and saved it to: ' + filename)
#        return ch4_xr
#    if trend:
#        ch4 = ch4_xr.trend
#        print_saved_file('trend ch4_index.nc', savepath)
#    else:
    ch4 = ch4_xr.trend
    if interpolate:
        dt = pd.date_range(start='1979-01-01', end='2019-12-01', freq='MS')
        ch4 = ch4.interp(time=dt)
        ch4 = ch4.interpolate_na(dim='time', method='spline')
    if savepath is not None:
        ch4.to_netcdf(savepath / 'ch4_index.nc', 'w')
        print_saved_file('ch4_index.nc', savepath)
    return ch4


def _produce_CH4_jaxa(load_path, savepath=None):
    import pandas as pd
    """http://www.gosat.nies.go.jp/en/assets/whole-atmosphere-monthly-mean_ch4_dec2019.zip"""
    df = pd.read_csv(
        load_path /
        '10x60.trend.method2.txt',
        comment='#',
        header=None, delim_whitespace=True)
    df.columns = ['year', 'month', 'mm', 'trend']
    idx = pd.to_datetime(dict(year=df.year, month=df.month,
                              day='01'))
    df = df.set_index(idx)
    df.index.name = 'time'
    df = df.drop(['year', 'month'], axis=1)
    ds = df.to_xarray() * 1000.0
    for da in ds.data_vars:
        ds[da].attrs['unit'] = 'ppb'
    return ds


def _produce_cpt_swoosh(load_path=work_chaim, savepath=None):
    import xarray as xr
    import pandas as pd
    sw = xr.open_dataset(load_path /
                         'swoosh-v02.6-198401-201812/swoosh-v02.6-198401-201812-latpress-2.5deg-L31.nc', decode_times=False)
    time = pd.date_range('1984-01-01', freq='MS', periods=sw.time.size)
    sw['time'] = time
    # cold point tropopause:
    cpt = sw['cptropt']
    cpt = cpt.sel(lat=slice(-15, 15))
    cpt = cpt.mean('lat')
    if savepath is not None:
        cpt.to_netcdf(savepath / 'cpt_index.nc')
        print_saved_file('cpt_index.nc', savepath)
    return cpt


def _produce_cpt_sean_ERA5(load_path=work_chaim/'Sean - tropopause', savepath=None):
    import xarray as xr
    import pandas as pd
    from aux_functions_strat import lat_mean
    from aux_functions_strat import anomalize_xr
    cpt = xr.load_dataset(load_path/'era5.tp.monmean.zm.nc')['ctpt']
    cpt = cpt.sel(lat=slice(15, -15))
    # attrs = cpt.attrs
    cpt = lat_mean(cpt)
    cpt.attrs['data from'] = 'ERA5'
    cpt['time'] = pd.to_datetime(cpt['time'].values).to_period('M').to_timestamp()
    cpt = anomalize_xr(cpt, freq='MS')
    if savepath is not None:
        cpt.to_netcdf(savepath / 'cpt_ERA5_index.nc')
    return cpt

#def _produce_cold_point(savepath=None, lonslice=None):
#    import xarray as xr
#    import sys
#    import os
#    # lonslice is a two-tuple : (minlon, maxlon)
#    if savepath is None:
#        savepath = os.getcwd() + '/regressors/'
#    if sys.platform == 'linux':
#        work_path = '/home/shlomi/Desktop/DATA/Work Files/Chaim_Stratosphere_Data/'
#    elif sys.platform == 'darwin':  # mac os
#        work_path = '/Users/shlomi/Documents/Chaim_Stratosphere_Data/'
#    era5 = xr.open_dataarray(work_path + 'ERA5_T_eq_all.nc')
#    if lonslice is None:
#        # cold_point = era5.sel(level=100).quantile(0.1, ['lat',
#        #                                                'lon'])
#        cold_point = era5.sel(level=100)
#        cold_point = cold_point.mean('lon')
#        cold_point = cold_point.mean('lat')
#        # cold_point = cold_point.rolling(time=3).mean()
#
#        # cold_point = era5.sel(level=slice(150, 50)).min(['level', 'lat',
#        #                                                  'lon'])
#    else:
#        # cold_point = era5.sel(level=100).sel(lon=slice(*lonslice)).quantile(
#        #    0.1, ['lat', 'lon'])
#        cold_point = era5.sel(level=slice(150, 50)).sel(
#                lon=slice(*lonslice)).min(['level', 'lat', 'lon'])
#        cold_point.attrs['lon'] = lonslice
#    cold_point.name = 'cold'
#    cold_point.to_netcdf(savepath + 'cold_point_index.nc')
#    print('Saved cold_point_index.nc to ' + savepath)
#    return cold_point
def _produce_CDAS_QBO(savepath=None):
    import pandas as pd
    url = 'https://www.cpc.ncep.noaa.gov/data/indices/qbo.u50.index'
    df = pd.read_csv(url, header=2, delim_whitespace=True)
    anom_index = df[df['YEAR'] == 'ANOMALY'].index.values.item()
    orig = df.iloc[0:anom_index - 2, :]
    stan_index = df[df['YEAR'] == 'STANDARDIZED'].index.values.item()
    anom = df.iloc[anom_index + 2: stan_index - 2, :]
    stan = df.iloc[stan_index + 2:-1, :]
    dfs = []
    for df in [orig, anom, stan]:
        df = df.head(42)  # keep all df 1979-2020
        # df.drop(df.tail(1).index, inplace=True)
        df = df.melt(id_vars='YEAR', var_name='MONTH')
        datetime = pd.to_datetime((df.YEAR + '-' + df.MONTH).apply(str), format='%Y-%b')
        df.index = datetime
        df = df.sort_index()
        df = df.drop(['YEAR', 'MONTH'], axis=1)
        df['value'] = df['value'].astype(float)
        dfs.append(df)
    all_df = pd.concat(dfs, axis=1)
    all_df.columns = ['original', 'anomaly', 'standardized']
    all_df.index.name='time'
    qbo = all_df.to_xarray()
    qbo.attrs['name'] = 'qbo_cdas'
    qbo.attrs['long_name'] = 'CDAS 50 mb zonal wind index'
    qbo['standardized'].attrs = qbo.attrs
    if savepath is not None:
        qbo.to_netcdf(savepath / 'qbo_cdas_index.nc')
        print_saved_file('qbo_cdas_index.nc', savepath)
    return qbo


def _produce_CO2(loadpath, filename='co2.txt'):
    import requests
    import io
    import xarray as xr
    import pandas as pd
    from aux_functions_strat import save_ncfile
    # TODO: complete this:
    filepath = loadpath / filename
    if filepath.is_file():
        print('co2 index already d/l and saved!')
        co2 = xr.open_dataset(filepath)
    else:
        print('Downloading CO2 index data from cpc website...')
        url = 'https://www.esrl.noaa.gov/gmd/webdata/ccgg/trends/co2/co2_mm_mlo.txt'
        s = requests.get(url).content
        co2_df = pd.read_csv(io.StringIO(s.decode('utf-8')),
                             delim_whitespace=True, comment='#')
        co2_df.columns = ['year', 'month', 'decimal_date', 'monthly_average', 'deseasonalized', 'days', 'days_std', 'mm_uncertainty']
        co2_df['dt'] = pd.to_datetime(co2_df['year'].astype(str) + '-' + co2_df['month'].astype(str))
        co2_df = co2_df.set_index('dt')
        co2_df.index.name = 'time'
        co2 = co2_df[['monthly_average', 'mm_uncertainty']].to_xarray()
        co2 = co2.rename(
            {'monthly_average': 'co2', 'mm_uncertainty': 'co2_error'})
        co2.attrs['name'] = 'CO2 index'
        co2.attrs['source'] = url
        co2['co2'].attrs['units'] = 'ppm'
        save_ncfile(co2, loadpath, 'co2_index.nc')
    return co2


def _produce_GHG(loadpath, savepath=None):
    import xarray as xr
    import numpy as np
    import pandas as pd
    from pathlib import Path
    aggi = pd.read_csv(loadpath / 'AGGI_Table.csv', index_col='Year', header=2)
    aggi = aggi[:-3]
    ghg = aggi.loc[:, '1990 = 1']
    ghg.name = 'GHG-RF'
    ghg.index = pd.to_datetime(ghg.index, infer_datetime_format=True)
    ghg_m = ghg.resample('MS').interpolate()
    # extend the index :
    ghg_m = pd.DataFrame(data=ghg_m,
                         index=pd.date_range(start=ghg_m.index[0],
                                             end='2018-09-01',
                                             freq=ghg_m.index.freq))
    # fit data:
    di = ghg_m.index
    df = ghg_m.reset_index().drop('index', 1)
    fit_df = df.dropna()
    fit = np.polyfit(fit_df.index.values, fit_df.values, 3)
    extp_func = np.poly1d(np.squeeze(fit))
    # extrapolate:
    nans_x = pd.isnull(df).any(1).nonzero()[0]
    Y = np.expand_dims(extp_func(nans_x), 1)
    df.loc[nans_x] = Y
    df.index = di
    ghg = xr.DataArray(np.squeeze(df), dims='time')
    if savepath is not None:
        ghg.to_netcdf(savepath / 'ghg_index.nc')
        print_saved_file('ghg_index.nc', savepath)
    return ghg


def _produce_OLR(loadpath, savepath=None):
    import xarray as xr
    import numpy as np
    import pandas as pd
    from pathlib import Path
    olr = xr.open_dataset(loadpath / 'olr-monthly_v02r07_197901_201901.nc',
                          decode_times=False)
    olr['time'] = pd.date_range('1979-01-01', '2019-01-01', freq='MS')
    olr = olr.mean('lon', keep_attrs=True)
    olr = olr.sel(lat=slice(-20, 20))
    olr['cos_lat'] = np.cos(np.deg2rad(olr['lat']))
    olr['olr_mean'] = (olr.cos_lat * olr.olr).sum('lat', keep_attrs=True) / \
        olr.cos_lat.sum('lat', keep_attrs=True)
    olr_da = olr.olr_mean
    olr_da.attrs = olr.olr.attrs
    if savepath is not None:
        olr_da.to_netcdf(savepath / 'olr_index.nc')
        print_saved_file('olr_index.nc', savepath)
    return olr_da


def _produce_T500_from_era5(loadpath, savepath=None):
    """  """
    # import os
    import xarray as xr
    from aux_functions_strat import lat_mean
    from aux_functions_strat import xr_rename_sort
    t500 = xr.open_dataarray(loadpath / 'era5_t500_mm_1979-2019.nc')
    t500 = xr_rename_sort(t500)
    t500 = t500.mean('lon')
    t500 = lat_mean(t500.sel(lat=slice(-20, 20)))
    if savepath is not None:
        t500.to_netcdf(savepath / 'era5_t500_index.nc')
        print_saved_file('era5_t500_index.nc', savepath)
    return t500


# def _produce_qbo_berlin()

def _produce_eof_pcs(loadpath, npcs=2, name='qbo', source='singapore',
                     levels=(100, 10), plot=True, savepath=None):
    import xarray as xr
    import aux_functions_strat as aux
    from eofs.xarray import Eof
    import matplotlib.pyplot as plt
    import numpy as np
    from pathlib import Path
    # load and order data dims for eofs:
    if source == 'singapore':
        U = _download_singapore_qbo(path=loadpath)
        U = aux.xr_order(U)
        # get rid of nans:
        U = U.sel(level=slice(90, 10))
        U = U.dropna(dim='time')
        filename = 'singapore_qbo_index.nc'
    elif source == 'era5':
        U = xr.open_dataarray(loadpath / 'ERA5_U_eq_mean.nc')
        U = U.sel(level=slice(100, 10))
        # U = U.sel(time=slice('1987', '2018'))
        filename = 'era5_qbo_index.nc'
    elif source == 'swoosh':
        U = xr.open_dataset(loadpath / 'swoosh_latpress-2.5deg.nc')
        U = U['combinedanomfillanomh2oq']
        U = U.sel(lat=slice(-20, 20), level=slice(*levels))
        U = aux.xr_weighted_mean(U)
        filename = 'swoosh_h2o_index.nc'
    solver = Eof(U)
    eof = solver.eofsAsCorrelation(neofs=npcs)
    pc = solver.pcs(npcs=npcs, pcscaling=1)
    pc.attrs['long_name'] = source + ' ' + name + ' index'
    pc['mode'] = pc.mode + 1
    eof['mode'] = eof.mode + 1
    vf = solver.varianceFraction(npcs)
    errors = solver.northTest(npcs, vfscaled=True)
    [name + '_' + str(i) for i in pc]
    qbo_ds = xr.Dataset()
    for ar in pc.groupby('mode'):
        qbo_ds[name + '_' + str(ar[0])] = ar[1]
    if source == 'era5':
        qbo_ds = -qbo_ds
    qbo_ds = qbo_ds.reset_coords(drop=True)
    if savepath is not None:
        qbo_ds.to_netcdf(savepath / filename, 'w')
        print_saved_file(filename, savepath)
    if plot:
        plt.close('all')
        plt.figure(figsize=(8, 6))
        eof.plot(hue='mode')
        plt.figure(figsize=(10, 4))
        pc.plot(hue='mode')
        plt.figure(figsize=(8, 6))
        x = np.arange(1, len(vf.values) + 1)
        y = vf.values
        ax = plt.gca()
        ax.errorbar(x, y, yerr=errors.values, color='b', linewidth=2, fmt='-o')
        ax.set_xticks(np.arange(1, len(vf.values) + 1, 1))
        ax.set_yticks(np.arange(0, 1, 0.1))
        ax.grid()
        ax.set_xlabel('Eigen Values')
        plt.show()
    return qbo_ds


#def download_enso_MEI(path='/Users/shlomi/Dropbox/My_backup/Python_projects/Stratosphere_Chaim/',
#                      filename='enso_MEI.nc'):
#    import os.path
#    import io
#    import pandas as pd
#    import xarray as xr
#    import numpy as np
#    if os.path.isfile(os.path.join(path, filename)):
#        print('NOAA ENSO MEI already d/l and saved!')
#        # read it to data array (xarray)
#        nino_xr = xr.open_dataset(path + filename)
#        # else d/l the file and first read it to df (pandas), then to xarray then save as nc:
#    else:
#        print('Downloading ENSO MEI data from noaa esrl website...')
#        url = 'https://www.esrl.noaa.gov/psd/enso/mei/table.html'
#        nino_df = pd.read_html(url)
#        # idx = pd.to_datetime(dict(year=nino_df.YR, month=nino_df.MON, day='1'))
#        # nino_df = nino_df.set_index(idx)
#        # nino_df = nino_df.drop(nino_df.iloc[:, 0:2], axis=1)
#        # nino_df.columns = ['NINO1+2', 'ANOM_NINO1+2', 'NINO3', 'ANOM_NINO3',
#        #                 'NINO4', 'ANOM_NINO4', 'NINO3.4', 'ANOM_NINO3.4']
#        # nino_df = nino_df.rename_axis('time')
#        # nino_xr = xr.Dataset(nino_df)
#        # nino_xr.to_netcdf(path + filename)
#        print('Downloaded NOAA ENSO MEI data and saved it to: ' + filename)
#    return nino_df

def _download_solar_10p7cm_flux(loadpath, filename='solar_10p7cm.nc',
                                savepath=None, index=False):
    """download the solar flux from Dominion Radio astrophysical Observatory
    Canada"""
    import ftputil
    import pandas as pd
    import xarray as xr
    from pathlib import Path
    filepath = loadpath / filename
    if filepath.is_file():
        print('Solar flux 10.7cm from DRAO Canada already d/l and saved!')
        # read it to data array (xarray)
        solar_xr = xr.open_dataset(path / filename)
        # else d/l the file and fObsirst read it to df (pandas),
        # then to xarray then save as nc:
    else:
        filename_todl = 'solflux_monthly_average.txt'
        with ftputil.FTPHost('ftp.geolab.nrcan.gc.ca', 'anonymous', '') as ftp_host:
            ftp_host.chdir('/data/solar_flux/monthly_averages/')
            ftp_host.download(filename_todl, path + filename_todl)
        solar_df = pd.read_csv(path / filename_todl, delim_whitespace=True,
                               skiprows=1)
        print('Downloading solar flux 10.7cm from DRAO Canada website...')
        idx = pd.to_datetime(dict(year=solar_df.Year, month=solar_df.Mon,
                                  day='1'))
        solar_df = solar_df.set_index(idx)
        solar_df = solar_df.drop(solar_df.iloc[:, 0:2], axis=1)
        solar_df = solar_df.rename_axis('time')
        solar_xr = xr.Dataset(solar_df)
        solar_xr.attrs['long_name'] = 'Monthly averages of Solar 10.7 cm flux'
        if savepath is not None:
            solar_xr.to_netcdf(savepath / filename)
            print('Downloaded solar flux 10.7cm data and saved it to: ' + filename)
    if index:
        solar = solar_xr.Adjflux
        solar.attrs['long_name'] = 'Solar Adjflux 10.7cm'
        if savepath is not None:
            solar.to_netcdf(savepath / 'solar_10p7cm_index.nc')
            print_saved_file('solar_10p7cm_index.nc', savepath)
        return solar
    else:
        return solar_xr


def _produce_strato_aerosol(loadpath, savepath=None, index=False,
                            filename='multiple_input4MIPs_aerosolProperties_CMIP_IACETH-SAGE3lambda-3-0-0_gn_185001_201412.nc'):
    import os.path
    import xarray as xr
    import numpy as np
    import pandas as pd
    from pathlib import Path
    from datetime import date, timedelta
    filepath = loadpath / filename
    if filepath.is_file():
        aerosol_xr = xr.open_dataset(loadpath / filename, decode_times=False)
        start_date = date(1850, 1, 1)
        days_from = aerosol_xr.time.values.astype('O')
        offset = np.empty(len(days_from), dtype='O')
        for i in range(len(days_from)):
            offset[i] = (start_date + timedelta(days_from[i])).strftime('%Y-%m')
        aerosol_xr['time'] = pd.to_datetime(offset)
        print('Importing ' + str(loadpath) + filename + ' to Xarray')
    else:
        print('File not found...')
        return
    if index:
        from aux_functions_strat import xr_weighted_mean
        vol = aerosol_xr.sad.sel(altitude=20)
        vol = vol.rename({'latitude': 'lat'})
        vol = xr_weighted_mean(vol.sel(lat=slice(-20, 20)))
        vol.attrs['long_name'] = 'Stratoapheric aerosol density'
        if savepath is not None:
            vol.to_netcdf(savepath / 'vol_index.nc')
            print_saved_file('vol_index.nc', savepath)
        return vol
    else:
        return aerosol_xr


#def download_nao(path='/Users/shlomi/Dropbox/My_backup/Python_projects/Stratosphere_Chaim/',
#                 filename='noaa_nao.nc'):
#    import requests
#    import os.path
#    import io
#    import pandas as pd
#    import xarray as xr
#    import numpy as np
#    if os.path.isfile(os.path.join(path, filename)):
#        print('Noaa NAO already d/l and saved!')
#        # read it to data array (xarray)
#        nao_xr = xr.open_dataarray(path + filename)
#        # else d/l the file and first read it to df (pandas), then to xarray then save as nc:
#    else:
#        print('Downloading nao data from noaa ncep website...')
#        url = 'http://www.cpc.ncep.noaa.gov/products/precip/CWlink/pna/norm.nao.monthly.b5001.current.ascii'
#        s = requests.get(url).content
#        nao_df = pd.read_csv(io.StringIO(s.decode('utf-8')), header=None, delim_whitespace=True)
#        nao_df.columns = ['YR', 'MON', 'nao']
#        idx = pd.to_datetime(dict(year=nao_df.YR, month=nao_df.MON, day='1'))
#        nao_df = nao_df.set_index(idx)
#        nao_df = nao_df.drop(nao_df.iloc[:, 0:2], axis=1)
#        nao_df = nao_df.rename_axis('time')
#        nao_df = nao_df.squeeze(axis=1)
#        nao_xr = xr.DataArray(nao_df)
#        nao_xr.attrs['long_name'] = 'North Atlantic Oscillation'
#        nao_xr.name = 'NAO'
#        nao_xr.to_netcdf(path + filename)
#        print('Downloaded nao data and saved it to: ' + filename)
#    return nao_xr

def _download_MJO_from_cpc(loadpath, filename='mjo.nc', savepath=None):
    import requests
    import io
    import xarray as xr
    import pandas as pd
    from aux_functions_strat import save_ncfile
    # TODO: complete this:
    filepath = loadpath / filename
    if filepath.is_file():
        print('MJO index already d/l and saved!')
        mjo = xr.open_dataset(filepath)
    else:
        print('Downloading MJO index data from cpc website...')
        url = 'https://www.cpc.ncep.noaa.gov/products/precip/CWlink/daily_mjo_index/proj_norm_order.ascii'
        s = requests.get(url).content
        mjo_df = pd.read_csv(io.StringIO(s.decode('utf-8')),
                              delim_whitespace=True, na_values='*****',
                              header=1)
        mjo_df['dt'] = pd.to_datetime(mjo_df['PENTAD'], format='%Y%m%d')
        mjo_df = mjo_df.set_index('dt')
        mjo_df = mjo_df.drop('PENTAD', axis=1)
        mjo_df.index.name = 'time'
        mjo = mjo_df.to_xarray()
        mjo.attrs['name'] = 'MJO index'
        mjo.attrs['source'] = url
        save_ncfile(mjo, loadpath, 'mjo.nc')
    return mjo


def _read_MISO_index(loadpath, filename='miso.nc', savepath=None):
    from aux_functions_strat import path_glob
    from aux_functions_strat import save_ncfile
    import pandas as pd
    files = path_glob(loadpath, '*_pc_MJJASO.dat')
    dfs = []
    for file in files:
        df = pd.read_csv(file, delim_whitespace=True, header=None)
        df.columns=['days_begining_at_MAY1', 'miso1', 'miso2', 'phase']
        year = file.as_posix().split('/')[-1].split('_')[0]
        dt = pd.date_range('{}-05-01'.format(year),
                           periods=df['days_begining_at_MAY1'].iloc[-1], freq='d')
        df = df.set_index(dt)
        dfs.append(df)
    dff = pd.concat(dfs, axis=0)
    dff = dff.sort_index()
    dff = dff.drop('days_begining_at_MAY1', axis=1)
    dff.index.name = 'time'
    miso = dff.to_xarray()
    miso.attrs['name'] = 'MISO index'
    save_ncfile(miso, loadpath, 'miso.nc')
    return miso


def _read_all_indian_rain_index(
        loadpath, filename='all_indian_rain.nc', savepath=None):
    import pandas as pd
    from aux_functions_strat import save_ncfile
    df = pd.read_csv(
        loadpath /
        'all_indian_rain_1871-2016.txt',
        delim_whitespace=True)
    df = df.drop(['JF', 'MAM', 'JJAS', 'OND', 'ANN'],axis=1)
    # transform from table to time-series:
    df = df.melt(id_vars='YEAR', var_name='month', value_name='rain')
    df['date'] = pd.to_datetime(df['YEAR'].astype(str) + '-' + df['month'])
    df.set_index('date', inplace=True)
    df = df.drop(['YEAR', 'month'], axis=1)
    df = df.sort_index()
    df.index.name = 'time'
    indian = df.to_xarray()
    indian.attrs['name'] = 'All indian rain index'
    save_ncfile(indian, loadpath, 'indian_rain.nc')
    return indian


def _download_enso_ersst(loadpath, filename='noaa_ersst_nino.nc', index=False,
                         savepath=None):
    import requests
    import os.path
    import io
    import pandas as pd
    import xarray as xr
    from pathlib import Path
    filepath = loadpath / filename
    if filepath.is_file():
        print('Noaa Ersst El-Nino SO already d/l and saved!')
        # read it to data array (xarray)
        nino_xr = xr.open_dataset(filepath)
        # else d/l the file and first read it to df (pandas),
        # then to xarray then save as nc:
    else:
        print('Downloading ersst nino data from noaa ncep website...')
        url = 'http://www.cpc.ncep.noaa.gov/data/indices/ersst5.nino.mth.81-10.ascii'
        url2 = 'https://www.cpc.ncep.noaa.gov/data/indices/ersst5.nino.mth.91-20.ascii'
        s = requests.get(url2).content
        nino_df = pd.read_csv(io.StringIO(s.decode('utf-8')),
                              delim_whitespace=True)
        idx = pd.to_datetime(dict(year=nino_df.YR, month=nino_df.MON, day='1'))
        nino_df = nino_df.set_index(idx)
        nino_df = nino_df.drop(nino_df.iloc[:, 0:2], axis=1)
        nino_df.columns = ['NINO1+2', 'ANOM_NINO1+2', 'NINO3', 'ANOM_NINO3',
                           'NINO4', 'ANOM_NINO4', 'NINO3.4', 'ANOM_NINO3.4']
        nino_df = nino_df.rename_axis('time')
        nino_xr = xr.Dataset(nino_df)
        if savepath is not None:
            nino_xr.to_netcdf(savepath / filename)
            print('Downloaded ersst_nino data and saved it to: ' + filename)
    if index:
        enso = nino_xr['ANOM_NINO3.4']
        enso.attrs['long_name'] = enso.name
        if savepath is not None:
            enso.to_netcdf(savepath / 'anom_nino3p4_index.nc', 'w')
            print_saved_file('anom_nino3p4_index.nc', savepath)
        return enso
    else:
        return nino_xr


#def download_enso_sstoi(
#        path='/Users/shlomi/Dropbox/My_backup/Python_projects/Stratosphere_Chaim/',
#        filename='noaa_sstoi_nino.nc'):
#    import requests
#    import os.path
#    import io
#    import pandas as pd
#    import xarray as xr
#    import numpy as np
#    if os.path.isfile(os.path.join(path, filename)):
#        print('Noaa Sstoi El-Nino SO already d/l and saved!')
#        # read it to data array (xarray)
#        nino_xr = xr.open_dataset(path + filename)
#        # else d/l the file and first read it to df (pandas), then to xarray then save as nc:
#    else:
#        print('Downloading sstoi nino data from noaa ncep website...')
#        url = 'http://www.cpc.ncep.noaa.gov/data/indices/sstoi.indices'
#        s = requests.get(url).content
#        nino_df = pd.read_csv(io.StringIO(s.decode('utf-8')),
#                              delim_whitespace=True)
#        idx = pd.to_datetime(dict(year=nino_df.YR, month=nino_df.MON, day='1'))
#        nino_df = nino_df.set_index(idx)
#        nino_df = nino_df.drop(nino_df.iloc[:, 0:2], axis=1)
#        ni(times=None, return_anom=False, plot=False, return_mean=True)no_df.columns = ['NINO1+2', 'ANOM_NINO1+2', 'NINO3', 'ANOM_NINO3',
#                           'NINO4', 'ANOM_NINO4', 'NINO3.4', 'ANOM_NINO3.4']
#        nino_df = nino_df.rename_axis('time')
#        nino_xr = xr.Dataset(nino_df)
#        nino_xr.to_netcdf(path + filename)
#        print('Downloaded sstoi_nino data and saved it to: ' + filename)
#    return nino_xr


#def download_solar_250nm(filename='nrl2_ssi.nc'):
#    import requests
#    import os.path
#    import io
#    import pandas as pd
#    import xarray as xr
#    import numpy as np
#    import os
#    from datetime import date, timedelta
#    path = os.getcwd() + '/'
#    if os.path.isfile(os.path.join(path, filename)):
#        print('Solar irridiance 250nm already d/l and saved!')
#        # (times=None, return_anom=False, plot=False, return_mean=True)read it to data array (xarray)
#        solar_xr = xr.open_dataarray(path + filename)
#        # else d/l the file and first read it to df (pandas), then to xarray then save as nc:
#    else:
#        print('Downloading solar 250nm irridiance data from Lasp Interactive Solar IRridiance Datacenter (LISIRD)...')
#        url = 'http://lasp.colorado.edu/lisird/latis/nrl2_ssi_P1M.csv?time,wavelength,irradiance&wavelength=249.5'
#        s = requests.get(url).content
#        solar_df = pd.read_csv(io.StringIO(s.decode('utf-8')))
#        start_date = date(1610, 1, 1)
#        days_from = solar_df.iloc[:, 0].values
#        offset = np.empty(len(days_from), dtype='O')
#        for i in range(len(days_from)):
#            offset[i] = (start_date + timedelta(days_from[i])).strftime('%Y-%m')
#        solar_df = solar_df.set_index(pd.to_datetime(offset))
#        solar_df = solar_df.drop(solar_df.iloc[:, 0:2], axis=1)
#        solar_df = solar_df.rename_axis('time').rename_axis('irradiance', axis='columns')
#        solar_xr = xr.DataArray(solar_df)
#        solar_xr.irradiance.attrs = {'long_name': 'Irradiance',
#          (times=None, return_anom=False, plot=False, return_mean=True)                           'units': 'W/m^2/nm'}
#        solar_xr.attrs = {'long_name': 'Solar Spectral Irradiance (SSI) at 249.5 nm wavelength from LASP'}
#        solar_xr.name = 'Solar UV'
#        solar_xr.to_netcdf(path + filename)
#        print('Downloaded ssi_250nm data and saved it to: ' + filename)
#    return solar_xr


def _download_singapore_qbo(path=None, filename='singapore_qbo_index.nc'):
    import requests
    import os.path
    import io
    import pandas as pd
    import xarray as xr
    import functools
    from pathlib import Path
    """checks the files for the singapore qbo index from Berlin Uni. and
    reads them or downloads them if they are
    missing. output is the xarray and csv backup locally"""
    if path is None:
        path = Path().cwd() / 'regressors/'
    filepath = path / filename
    if filepath.is_file():
        print('singapore QBO already d/l and saved!')
        # read it to data array (xarray)
        sqbo_xr = xr.open_dataset(path / filename)
        # else d/l the file and first read it to df (pandas),
        # then to xarray then save as nc:
    else:
        print('Downloading singapore data from Berlin university...')
        url = 'http://www.geo.fu-berlin.de/met/ag/strat/produkte/qbo/singapore.dat'
        s = requests.get(url).content
        sing_qbo = sing_qbo = pd.read_csv(io.StringIO(s.decode('utf-8')),
                                          skiprows=3,
                                          header=None, delim_whitespace=True,
                                          names=list(range(0, 13)))
        # take out year data
        year = sing_qbo.iloc[0:176:16][0]
        # from 1997 they added another layer (100 hPa) to data hence the
        # irregular indexing:
        year = pd.concat([year, sing_qbo.iloc[177::17][0]], axis=0)
        df_list = []
        # create a list of dataframes to start assembeling data:
        for i in range(len(year)-1):
            df_list.append(pd.DataFrame(
                    data=sing_qbo.iloc[year.index[i] + 1:year.index[i + 1]]))
        # don't forget the last year:
        df_list.append(pd.DataFrame(data=sing_qbo.iloc[year.index[-1]+1::]))
        for i in range(len(df_list)):
            # first reset the index:
            df_list[i] = df_list[i].reset_index(drop=True)
            # drop first row:
            df_list[i] = df_list[i].drop(df_list[i].index[0], axis=0)
            # change all data to float:
            df_list[i] = df_list[i].astype('float')
            # set index to first column (hPa levels):
            df_list[i].set_index(0, inplace=True)
            # multiply all values to  X 0.1 (scale factor from website)
            df_list[i] = df_list[i] * 0.1
            # assemble datetime index and apply to df.columns:
            first_month = year.iloc[i] + '-01-01'
            time = pd.date_range(first_month, freq='MS', periods=12)
            df_list[i].columns = time
            # df_list[i].rename_axis('hPa').rename_axis('time', axis='columns')
        # do an outer join on all df_list
        df_combined = functools.reduce(
                lambda df1, df2: df1.join(
                        df2, how='outer'), df_list)
        df_combined = df_combined.rename_axis(
                'level').rename_axis('time', axis='columns')
        df_combined = df_combined.sort_index(ascending=False)
        sqbo_xr = xr.DataArray(df_combined)
        sqbo_xr.level.attrs = {'long_name': 'pressure',
                               'units': 'hPa',
                               'positive': 'down',
                               'axis': 'Z'}
        sqbo_xr.attrs = {'long_name': 'Monthly mean zonal wind components' +
                         ' at Singapore (48698), 1N/104E',
                         'units': 'm/s'}
        sqbo_xr.name = 'Singapore QBO'
        sqbo_xr.to_netcdf(path / filename)
        print('Downloaded singapore qbo data and saved it to: ' + filename)
    return sqbo_xr


def _produce_BDC(loadpath, plevel=70, savepath=None):
#    from era5_tools import proccess_era5_fields
#    bdc = proccess_era5_fields(path=loadpath, pre_names='MTTPM_54',
#                               post_name='bdc_index', mean=True,
#                               savepath=savepath)
    from aux_functions_strat import path_glob
    from aux_functions_strat import lat_mean
    import xarray as xr
    file = path_glob(loadpath, 'era5_mttpm_{}hPa.nc'.format(plevel))[0]
    da = xr.load_dataarray(file)
    bdc = lat_mean(da.sel(lat=slice(-5, 5)))
    bdc = bdc.mean('lon', keep_attrs=True).squeeze(drop=True)
    filename = 'era5_bdc{}_index.nc'.format(plevel)
    if savepath is not None:
        bdc.to_netcdf(savepath / filename, 'w')
        print_saved_file(filename, savepath)
    return bdc


def _produce_radio_cold(savepath=None, no_qbo=False, rolling=None):
    from strato_soundings import calc_cold_point_from_sounding
    from aux_functions_strat import overlap_time_xr
    from sklearn.linear_model import LinearRegression
    from aux_functions_strat import deseason_xr
    import xarray as xr
    radio = calc_cold_point_from_sounding(
            times=None,
            return_anom=True,
            plot=False,
            return_mean=True)
    filename_prefix = 'radio_cold'
    if no_qbo:
        filename_prefix = 'radio_cold_no_qbo'
        # qbos = _produce_eof_pcs(reg_path, source='era5', plot=False)
        qbos = xr.open_dataset(savepath / 'qbo_cdas_index.nc')
        new_time = overlap_time_xr(qbos, radio)
        qbos = qbos.sel(time=new_time)
        qbos = deseason_xr(qbos.to_array('regressors'), how='mean').to_dataset('regressors')
        radio = radio.sel(time=new_time)
        lr = LinearRegression()
        X = qbos.to_array('regressors').T
        lr.fit(X, radio)
        radio = radio - lr.predict(X)
    if rolling is not None:
        filename_prefix += '_rolling{}'.format(rolling)
        radio = radio.rolling(time=rolling, center=False).mean()
    if savepath is not None:
        filename = filename_prefix + '_index.nc'
        radio.to_netcdf(savepath / filename, 'w')
        print_saved_file(filename, savepath)
    return radio


def _produce_totexttau(loadpath=work_chaim/'MERRA2/aerosol_carbon',
                       savepath=None, indian=False):
    import xarray as xr
    from aux_functions_strat import lat_mean
    ds = xr.load_dataset(loadpath / 'MERRA2_aerosol.nc')
    ds = ds.sortby('time')
    da = ds['TOTEXTTAU']
    if indian:
        da = da.sel(lat=slice(5, 30), lon=slice(65, 95))
        filename = 'merra2_aod_indian_index.nc'
    else:
        filename = 'merra2_aod_index.nc'
    da = lat_mean(da.mean('lon', keep_attrs=True))
    da = da.resample(time='MS').mean()
    if savepath is not None:
        da.to_netcdf(savepath / filename, 'w')
        print_saved_file(filename, savepath)
    return da


def _produce_nao(loadpath=reg_path, savepath=reg_path):
    import pandas as pd
    df = pd.read_csv(loadpath / 'nao.txt',
                     delim_whitespace=True,
                     names=[
                         'year',
                         'month',
                         'nao'])
    df['time'] = df['year'].astype(str) + '-' + df['month'].astype(str)

    df['time'] = pd.to_datetime(df['time'])
    df = df.set_index('time')
    df = df.drop(['year', 'month'], axis=1)
    da = df.to_xarray()
    if savepath is not None:
        filename = 'nao_index.nc'
        da.to_netcdf(savepath / filename)
        print_saved_file(filename, savepath)
    return da

def _download_NAO(savepath=reg_path):
    import pandas as pd
    import requests
    import io
    url = 'https://www.cpc.ncep.noaa.gov/products/precip/CWlink/pna/norm.nao.monthly.b5001.current.ascii'
    r = requests.get(url).content
    df = pd.read_csv(io.StringIO(r.decode('utf-8')),
                     delim_whitespace=True, comment='#')
    df.columns = ['year', 'month', 'nao_index']
    df['dt'] = pd.to_datetime(df['year'].astype(str) + '-' + df['month'].astype(str))
    df = df.set_index('dt')
    df.index.name = 'time'
    df = df.drop(['year','month'], axis=1)
    if savepath is not None:
        df.to_csv(savepath/'nao.txt')
    return df



def _produce_ea_wr(loadpath=reg_path, savepath=reg_path):
    import pandas as pd
    df = pd.read_csv(loadpath / 'EA-WR.txt',
                     delim_whitespace=True,
                     names=[
                         'year',
                         'month',
                         'ea-wr'])
    df['time'] = df['year'].astype(str) + '-' + df['month'].astype(str)

    df['time'] = pd.to_datetime(df['time'])
    df = df.set_index('time')
    df = df.drop(['year', 'month'], axis=1)
    da = df.to_xarray()
    if savepath is not None:
        filename = 'ea-wr_index.nc'
        da.to_netcdf(savepath / filename)
        print_saved_file(filename, savepath)
    return da


def _produce_pdo(loadpath=reg_path, savepath=reg_path):
    import pandas as pd
    df = pd.read_csv(loadpath / 'pdo.txt',
                     names=['date', 'pdo'], skiprows=2)
    df['time']=pd.to_datetime(df['date'],format='%Y%m')
    df = df.set_index('time')
    df = df.drop('date', axis=1)
    da = df.to_xarray()
    if savepath is not None:
        filename = 'pdo_index.nc'
        da.to_netcdf(savepath / filename)
        print_saved_file(filename, savepath)
    return da


def _produce_moi2(loadpath=reg_path, savepath=reg_path):
    import pandas as pd
    df = pd.read_fwf(
        reg_path /
        'moi2.txt',
        names=['year', 'date', 'moi2'],
        widths=[4, 8, 5])
    df['date'] = df['date'].str.strip('.')
    df['date'] = df['date'].str.strip(' ')
    df['date'] = df['date'].str.replace(' ', '0')
    df['date'] = df['date'].str.replace('.', '-')
    df['time'] = df['year'].astype(str) + '-' + df['date'].astype(str)
    df['time'] = pd.to_datetime(df['time'], format='%Y-%m-%d')
    df = df.set_index('time')
    df = df.drop(['date', 'year'], axis=1)
    da = df.to_xarray()
    if savepath is not None:
        filename = 'moi2_index.nc'
        da.to_netcdf(savepath / filename)
        print_saved_file(filename, savepath)
    return da


def _produce_mei_v1(loadpath=reg_path, savepath=reg_path):
    import pandas as pd
    df = pd.read_csv(loadpath / 'meiv1.txt',
                     names=['YEAR', 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                     skiprows=10, delim_whitespace=True)
    df = df.iloc[0:69]
    df = pd.melt(df, id_vars='YEAR', var_name='month', value_name='meiv1')
    df['month']=df['month'].astype(int)
    df['time'] = df['YEAR'].astype(str) + '-' + df['month'].astype(str)
    df['time'] = pd.to_datetime(df['time'])
    df = df.set_index('time')
    df = df.sort_index()
    df  = df.drop(['YEAR', 'month'], axis=1)
    df = df.astype(float)
    da = df.to_xarray()
    if savepath is not None:
        filename = 'meiv1_index.nc'
        da.to_netcdf(savepath / filename)
        print_saved_file(filename, savepath)
    return da


def _produce_mei_v2(loadpath=reg_path, savepath=reg_path):
    import pandas as pd
    df = pd.read_csv(loadpath / 'meiv2.txt',
                     names=['YEAR', 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                     skiprows=3, delim_whitespace=True, na_values=-999)
    df = pd.melt(df, id_vars='YEAR', var_name='month', value_name='meiv2')
    df['time'] = df['YEAR'].astype(str) + '-' + df['month'].astype(str)
    df['time'] = pd.to_datetime(df['time'])
    df = df.set_index('time')
    df = df.sort_index()
    df = df.drop(['YEAR', 'month'], axis=1)
    da = df.to_xarray()
    if savepath is not None:
        filename = 'meiv2_index.nc'
        da.to_netcdf(savepath / filename)
        print_saved_file(filename, savepath)
    return da


def _produce_glossac_aod(loadpath=work_chaim, savepath=reg_path):
    import xarray as xr
    import pandas as pd
    from aux_functions_strat import lat_mean
    glo = xr.load_dataset(work_chaim / 'GloSSAC_V2.0.nc')
    time = pd.to_datetime(glo['time'].values, format='%Y%m')
    glo['time'] = time
    glo['wavelengths_glossac'] = [str(x)
                                  for x in glo['wavelengths_glossac'].values]
    aod_wl = glo['Glossac_Aerosol_Optical_Depth'].to_dataset(
        'wavelengths_glossac')
    aod_wl = aod_wl.drop(['386', '452'])
    aod_eq = lat_mean(aod_wl.sel(lat=slice(-15, 15)))
    aod = aod_eq['525']
    aod.name = 'aod'
    if savepath is not None:
        filename = 'aod_index.nc'
        aod.to_netcdf(savepath / filename)
        print_saved_file(filename, savepath)
    return aod


def _produce_volcanic_aot(loadpath=reg_path, savepath=reg_path):
    import pandas as pd
    df = pd.read_csv(
        loadpath /
        'AOT.txt',
        skiprows=2,
        delim_whitespace=True,
        names=[
            'date',
            'aot'])
    df['year'] = df['date'].astype(int)
    df['month'] = (round((df['date'] - df['year']) * 12) + 1).astype(int)
    df['time'] = df['year'].astype(str) + '-' + df['month'].astype(str)
    df['time'] = pd.to_datetime(df['time'])
    df = df.set_index('time')
    df = df.drop(['date', 'year', 'month'], axis=1)
    da = df.to_xarray()
    if savepath is not None:
        filename = 'aot_index.nc'
        da.to_netcdf(savepath / filename)
        print_saved_file(filename, savepath)
    return da


def _produce_linear_trend(times=['1979', '2019'], savepath=reg_path):
    import pandas as pd
    import numpy as np
    import xarray as xr
    start = pd.to_datetime('{}-01-01'.format(times[0]))
    end = pd.to_datetime('{}-12-31'.format(times[1]))
    time = pd.date_range(start=start, end=end, freq='MS')
    trend=np.linspace(0, 1, num=len(time))
    da = xr.DataArray(trend, dims=['time'])
    da['time'] = time
    if savepath is not None:
        filename = 'trend_index.nc'
        da.to_netcdf(savepath / filename)
        print_saved_file(filename, savepath)
    return da
#def plot_time_over_pc_xr(pc, times, norm=5):
#    import numpy as np
#    import matplotlib.pyplot as plt
#    import aux_functions_strat as aux
#    min_times = times.time.values.min()
#    max_times = times.time.values.max()
#    min_pc = pc.time.values.min()
#    max_pc = pc.time.values.max()
#    max_time = np.array((max_times, max_pc)).min()
#    min_time = np.array((min_times, min_pc)).max()
#    fig, ax = plt.subplots(figsize=(16, 4))
#    pc = aux.normalize_xr(pc, norm)
#    times = aux.normalize_xr(times, norm)
#    pc.sel(time=slice(min_time, max_time)).plot(c='b', ax=ax)
#    times.sel(time=slice(min_time, max_time)).plot(c='r', ax=ax)
#    ax.set_title(pc.attrs['long_name'] + ' mode:' + str(pc.mode.values) +
#                 ' and ' + times.name)
#    ax.set_ylabel('')
#    plt.legend(['PC', times.name])
#    plt.show()
#    return
def create_nino_time_mask(loadpath=reg_path, thresh=0.5, event='la_nina',
                          times=None, plot=True):
    enso = _download_enso_ersst(loadpath)
    if event == 'la_nina':
        index = enso['ANOM_NINO3.4'].sel(time=slice('1984', '2019')).where(enso['ANOM_NINO3.4'] < -thresh)
        name = 'la_nina_events'
    elif event == 'el_nino':
        index = enso['ANOM_NINO3.4'].sel(time=slice('1984', '2019')).where(enso['ANOM_NINO3.4'] > thresh)
        name = 'el_nino_events'
    elif event is None:
        index = enso['ANOM_NINO3.4'].sel(time=slice('1984', '2019')).where(enso['ANOM_NINO3.4'] <= thresh).where(enso['ANOM_NINO3.4'] >= -thresh)
        name = 'neutral_enso'
    if times is not None:
        index = index.sel(time=slice(*times))
        print('Dates selected: {} to {}'.format(*times))
    da = index.dropna('time')['time']
    da.name = name
    if plot:
        index.plot()
    return da


def create_season_avg_nino(season='DJF'):
    import xarray as xr
    import pandas as pd
    ds = load_all_regressors()
    nino = ds['anom_nino3p4'].dropna('time')
    nino_djf=nino_djf=nino.sel(time=nino['time.season']=='DJF')
    df= nino_djf.to_dataframe()
    dfr = df.resample('Q-NOV').mean().dropna()
    start = '{}-12-01'.format(dfr.index.year.min() - 1)
    end = '{}-12-01'.format(dfr.index.year.max() - 1)
    new_time = pd.date_range(start=start, end=end, freq='12MS')
    dfr = dfr.set_index(new_time)
    new_time = pd.date_range(start=dfr.index.min(), end=dfr.index.max(), freq='MS')
    dfr = dfr.reindex(new_time).ffill()
    dfr.index.name = 'time'
    nino = dfr.to_xarray()['anom_nino3p4']
    return nino


def _make_nc_files_run_once(loadpath=work_chaim, savepath=None):
    from pathlib import Path
    if savepath is None:
        savepath = Path().cwd() / 'regressors/'
    _ = _download_enso_ersst(index=True, path=savepath)
    _ = _produce_strato_aerosol(loadpath, savepath=savepath, index=True)
    _ = _download_solar_10p7cm_flux(savepath=savepath, index=True)
    _ = _produce_eof_pcs(loadpath, npcs=2, name='qbo', source='singapore',
                         plot=False, savepath=savepath)
    _ = _produce_eof_pcs(loadpath, npcs=2, name='qbo', source='era5',
                         plot=False, savepath=savepath)
    _ = _produce_T500_from_era5(loadpath, savepath=savepath)
    _ = _download_CH4(trend=True, savepath=savepath)
    _ = _produce_OLR(savepath=savepath)
    _ = _produce_GHG(savepath=savepath)
    _ = _produce_wind_shear(source='singapore', savepath=savepath)
    _ = _produce_wind_shear(source='era5', savepath=savepath)
    _ = _produce_cpt_swoosh(savepath=savepath)
    _ = _produce_totexttau(savepath=savepath)
    # _ = _produce_BDC(loadpath, savepath=savepath)
    return
