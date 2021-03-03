#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 15:07:37 2021

@author: shlomi
"""
from strat_paths import work_chaim
mri_path = work_chaim/'MRI'


def load_a_b_p0(path=mri_path):
    import xarray as xr
    wv = xr.open_dataset(path/'vmrh2o_monthly_MRI-ESM1r1_refC2_r1i1p1_204001-204912.nc')
    a = wv['a']
    b = wv['b']
    p0 = wv['p0']
    return a, b, p0


def convert_hybrid_sigma_to_pressure(a, b, p0, ps):
    return a*p0 + b*ps


def extract_eq_var_at_pressure_level(path=mri_path, var='vmrh2o', plevel=8000):
    import xarray as xr
    from aux_functions_strat import path_glob
    from aux_functions_strat import save_ncfile
    file = path_glob(path, '{}_*equatorial*.nc'.format(var))[0]
    var_ds = xr.load_dataset(file)
    ps_file = path_glob(path, 'ps_*equatorial*.nc')[0]
    ps = xr.load_dataset(ps_file)['ps']
    a, b, p0 = load_a_b_p0(path=path)
    plev = convert_hybrid_sigma_to_pressure(a, b, p0, ps)
    var_dts = []
    for dt in plev.time:
        var_dt = var_ds.sel(time=dt)
        pl = plev.sel(time=dt)
        var_dt['lev'] = pl
        var_dt = var_dt.sel(lev=plevel, method='nearest')
        var_dts.append(var_dt)
    var_plev = xr.concat(var_dts, 'time')
    var_at_plevel = var_plev.reset_coords(drop=True)[var]
    var_at_plevel.attrs['plevel'] = str(plevel) + 'Pa'
    yrmax = var_at_plevel.time.max().dt.year.item()
    yrmin = var_at_plevel.time.min().dt.year.item()
    filename = '{}_equatorial_{}_Pa_{}-{}.nc'.format(var, plevel, yrmin, yrmax)
    save_ncfile(var_at_plevel, path, filename)
    return var_at_plevel


def produce_eq_means_from_3D_MRI(path=mri_path, var='vmrh2o', lat_slice=[-30, 30]):
    from aux_functions_strat import path_glob
    import xarray as xr
    from aux_functions_strat import save_ncfile
    from aux_functions_strat import lat_mean
    print('producing equatorial means for {}.'.format(var))
    filestr = 'monthly_MRI-ESM1r1_refC2_r1i1p1'
    files = sorted(path_glob(path, '{}_{}*.nc'.format(var, filestr)))
    dsl = [xr.open_dataset(x) for x in files]
    eqs = []
    for ds in dsl:
        eq_var = ds[var].mean('lon', keep_attrs=True)
        eq_var = lat_mean(eq_var.sel(lat=slice(*lat_slice)))
        eqs.append(eq_var)
    eq_da = xr.concat(eqs, 'time')
    eq_da = eq_da.sortby('time')
    yrmax = eq_da.time.max().dt.year.item()
    yrmin = eq_da.time.min().dt.year.item()
    filename = '{}_{}_equatorial_{}-{}.nc'.format(var, filestr, yrmin, yrmax)
    save_ncfile(eq_da, path, filename)
    return


def run_LR(path=mri_path, annual=False, times=['2000', None], detrend=False):
    from sklearn.linear_model import LinearRegression
    import xarray as xr
    from aux_functions_strat import anomalize_xr
    # load h2o:
    h2o = xr.load_dataarray(path / 'vmrh2o_equatorial_8000_Pa_1960-2099.nc')
    h2o = h2o.sel(time=slice(*times))
    # convert to ppmv:
    h2o *= 1e6
    h2o -= h2o.sel(time=slice('2000', '2009')).mean('time')
    # h2o /= h2o.sel(time=slice('2000', '2009')).std('time')
    # load t500:
    t500 = xr.load_dataarray(path / 'ta_equatorial_50000_Pa_1960-2099.nc')
    t500 = t500.sel(time=slice(*times))
    t500 -= t500.sel(time=slice('2000', '2009')).mean('time')
    # t500 /= t500.sel(time=slice('2000', '2009')).std('time')
    # load u50:
    u50 = xr.load_dataarray(path / 'ua_equatorial_5000_Pa_1960-2099.nc')
    # now produce qbo:
    u50 = u50.sel(time=slice(*times))
    u50 -= u50.sel(time=slice('2000', '2009')).mean('time')
    # u50 /= u50.sel(time=slice('2000', '2009')).std('time')
    # qbo = anomalize_xr(u50, 'MS', units='std')
    # detrend h2o and t500:
    if detrend:
        t500 = detrend_ts(t500)
        h2o = detrend_ts(h2o)
    # produce X, y:
    # y = anomalize_xr(h2o, 'MS', units='std')
    y = h2o
    X = xr.merge([t500, u50]).to_array('X')
    X = X.transpose('time', ...)
    if annual:
        y = y.resample(time='AS').mean()
        X = X.resample(time='AS').mean()
    # linear regresion:
    lr = LinearRegression()
    lr.fit(X, y)
    pred = xr.DataArray(lr.predict(X), dims=['time'])
    pred['time'] = y['time']
    print(lr.coef_)
    print(lr.score(X, y))
    y.plot(color='b')
    pred.plot(color='r')
    df=X.to_dataset('X').to_dataframe()
    df['h2o'] = y.to_dataframe()
    return lr, df


def detrend_ts(da_ts):
    trend = loess_curve(da_ts, plot=False)
    detrended = da_ts - trend['mean']
    detrended.name = da_ts.name
    return detrended


def loess_curve(da_ts, time_dim='time', season=None, plot=True):
    from skmisc.loess import loess
    import matplotlib.pyplot as plt
    import xarray as xr
    import numpy as np
    if season is not None:
        da_ts = da_ts.sel({time_dim: da_ts[time_dim + '.season'] == season})
    x = da_ts.dropna(time_dim)[time_dim].values
    y = da_ts.dropna(time_dim).values
    l_obj = loess(x, y)
    l_obj.fit()
    pred = l_obj.predict(x, stderror=True)
    conf = pred.confidence()
    lowess = np.copy(pred.values)
    ll = np.copy(conf.lower)
    ul = np.copy(conf.upper)
    da_lowess = xr.Dataset()
    da_lowess['mean'] = xr.DataArray(lowess, dims=[time_dim])
    da_lowess['upper'] = xr.DataArray(ul, dims=[time_dim])
    da_lowess['lower'] = xr.DataArray(ll, dims=[time_dim])
    da_lowess[time_dim] = x
    if plot:
        plt.plot(x, y, '+')
        plt.plot(x, lowess)
        plt.fill_between(x, ll, ul, alpha=.33)
        plt.show()
    return da_lowess
