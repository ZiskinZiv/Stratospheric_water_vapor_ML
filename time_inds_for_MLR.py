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
    
"""


def get_wind_shear(plot=True, source='singapore', deseason=True):
    import os
    import xarray as xr
    if source == 'singapore':
        u = get_singapore_qbo_save_locally()
    elif source == 'era5':
        u = xr.open_dataarray(os.getcwd() + '/' + 'ERA5_U_eq_mean.nc')
    wind_shear = u.diff('level').sel(level=70)
    if deseason:
        from aux_functions_strat import deseason_xr
        wind_shear = deseason_xr(wind_shear)
    wind_shear.name = 'wind_shear'
    if plot:
        wind_shear.plot()
    return wind_shear


def get_CH4(plot=True, filename='ch4_mm.nc', trend=True, deseason=True):
    import xarray as xr
    import pandas as pd
    import ftputil
    import os.path
    import os
    path = os.getcwd() + '/'
    if os.path.isfile(os.path.join(path, filename)):
        print('CH4 monthly means from NOAA ERSL already d/l and saved!')
        # read it to data array (xarray)
        ch4_xr = xr.open_dataset(path + filename)
        # else d/l the file and fObsirst read it to df (pandas), then to xarray then save as nc:
    else:
        filename_todl = 'ch4_mm_gl.txt'
        with ftputil.FTPHost('aftp.cmdl.noaa.gov', 'anonymous', '') as ftp_host:
            ftp_host.chdir('/products/trends/ch4/')
            ftp_host.download(filename_todl, path + filename_todl)
        ch4_df = pd.read_csv(path + filename_todl, delim_whitespace=True,
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
        ch4_xr.to_netcdf(path + filename)
        print('Downloaded CH4 monthly means data and saved it to: ' + filename)
    if plot:
        ch4_xr[['average', 'trend']].to_dataframe().plot()
    if trend:
        ch4 = ch4_xr.trend
    else:
        ch4 = ch4_xr.average
        if deseason:
            from aux_functions_strat import deseason_xr
            ch4 = deseason_xr(ch4)
    ch4.name = 'ch4'
    return ch4


def get_cold_point(plot=True, lonslice=None, deseason=True):
    import xarray as xr
    # import numpy as np
    era5 = xr.open_dataarray(work_path + 'ERA5_T_eq_all.nc')
    # cold_point = np.empty(era5.time.shape)
    # cold_point = [era5.isel(time=i).sel(level=slice(100, 80)).quantile(0.1) for
    #              i in range(len(era5.time.values))]
    if lonslice is None:
        # cold_point = era5.sel(level=100).quantile(0.1, ['lat',
        #                                                'lon'])
        cold_point = era5.sel(level=slice(150, 50)).min(['level', 'lat', 'lon'])
    else:
        # cold_point = era5.sel(level=100).sel(lon=slice(*lonslice)).quantile(
        #    0.1, ['lat', 'lon'])
        cold_point = era5.sel(level=slice(150, 50)).sel(lon=slice(*lonslice)).min(['level', 'lat', 'lon'])
        cold_point.attrs['lon'] = lonslice
    # cold_point = cold_point.sel(time=slice('1992-01','2002-08'))
    if deseason:
        from aux_functions_strat import deseason_xr
        cold_point = deseason_xr(cold_point)
    cold_point.name = 'cold'
    if plot:
        cold_point.plot()
    return cold_point


def get_GHG(plot=True):
    import os
    import xarray as xr
    import aux_functions_strat as aux
    import numpy as np
    import pandas as pd
    path = os.getcwd() + '/'
    aggi = pd.read_csv(path + 'AGGI_Table.csv', index_col='Year', header=2)
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
    if plot:
        ghg.plot()
    return ghg


def get_OLR(plot=True):
    import os
    import xarray as xr
    import aux_functions_strat as aux
    import numpy as np
    import pandas as pd
    path = os.getcwd() + '/'
    # olr = xr.open_dataset(path + 'olr.mon.mean.nc')
    # olr = olr.sortby('lat')
    # olr = olr.dropna('time')
    olr = xr.open_dataset(path + 'olr-monthly_v02r07_197901_201901.nc',
                          decode_times=False)
    olr['time'] = pd.date_range('1979-01-01', '2019-01-01', freq='MS')
    # olr = olr.drop(np.datetime64('1994-09-01'), 'time')
    olr = olr.mean('lon', keep_attrs=True)
    # olr = olr.sel(time=slice('1984-01-01', '2018-09-01'))
    olr = olr.sel(lat=slice(-20, 20))
    olr['cos_lat'] = np.cos(np.deg2rad(olr['lat']))
    olr['olr_mean'] = (olr.cos_lat * olr.olr).sum('lat', keep_attrs=True) / \
        olr.cos_lat.sum('lat', keep_attrs=True)
    olr_da = olr.olr_mean  # aux.xr_weighted_mean(olr.olr)
    olr_da.attrs = olr.olr.attrs
    if plot:
        olr_da.plot()
    return olr_da


def get_BDC(plot=True, deseason=True):
    """takes a 70hpa [-17 index on level var] for zonal_mean and slices -20 to 20 deg lat and saves it
    into a file. will return 4 time-series for each weighted mean selection :-20to20 -15 to 15 -10"""
    # import os
    import xarray as xr
    import aux_functions_strat as aux
    # path = os.getcwd() + '/'
    # bdc = xr.open_dataarray(path + 'MERRA_BDC70.nc')
    bdc = xr.open_dataarray(work_path + 'ERA5_BDC.nc')
    # bdc = bdc.sel(lat=slice(-20, 20))
    # bdc = aux.xr_weighted_mean(bdc)
    # BDC = xr.Dataset()
    # latmean = [5, 10, 15, 20, 25]
    # for lm in latmean:
    #     BDC['bdc' + str(lm)] = aux.xr_weighted_mean(bdc.sel(lat=slice(-lm, lm)))
    # BDC.attrs = bdc.attrs
    if deseason:
        from aux_functions_strat import deseason_xr
        bdc = deseason_xr(bdc)
    if plot:
        bdc.plot()
    return bdc


def get_T500(plot=True, deseason=True):
    """takes a 500 hpa [16 index on level var] from merra gcm temperature and slices -20 to 20 deg lat 
    saves it to file. retures 4 time-series for each weighted means sel"""
    # import os
    import xarray as xr
    import aux_functions_strat as aux
    import numpy as np
    t500 = xr.open_dataarray(work_path + 'ERA5_T_eq_all.nc')
    # t500 = t500.sel(lat=slice(-20, 20))
    t500 = t500.to_dataset(name='t500')
    t500 = t500.mean('lon')
    t500 = t500.sel(level=500)
    t500['cos_lat'] = np.cos(np.deg2rad(t500['lat']))
    t500['mean'] = (t500.cos_lat * t500.t500).sum('lat', keep_attrs=True) / \
           t500.cos_lat.sum('lat', keep_attrs=True)
    # t500 = aux.xr_weighted_mean(t500)
    # path = os.getcwd() + '/'
    # t500 = xr.open_dataarray(path + 'MERRA_T500.nc')
    # T500 = xr.Dataset()
    # latmean = [5, 10, 15, 20, 25]
    # for lm in latmean:
    #     T500['T' + str(lm)] = aux.xr_weighted_mean(t500.sel(lat=slice(-lm, lm)))
    # T500.attrs = t500.attrs
    T500 = t500['mean']
    if deseason:
        from aux_functions_strat import deseason_xr
        T500 = deseason_xr(T500)
    if plot:
        # T500.to_dataframe().plot()
        T500.plot()
    return T500


def get_qbo_pcs(npcs=2, source='singapore', plot=True):
    import os
    import xarray as xr
    import aux_functions_strat as aux
    from eofs.xarray import Eof
    import matplotlib.pyplot as plt
    import numpy as np
    # load and order data dims for eofs:
    if source == 'singapore':
        U = xr.open_dataarray(os.getcwd() + '/' + 'singapore_qbo.nc')
        U = aux.xr_order(U)
        # get rid of nans:
        U = U.sel(level=slice(90, 10))
        U = U.dropna(dim='time')
    elif source == 'era5':
        U = xr.open_dataarray(os.getcwd() + '/' + 'ERA5_U_eq_mean.nc')
        U = U.sel(level=slice(100, 10))
    solver = Eof(U)
    eof = solver.eofsAsCorrelation(neofs=npcs)
    pc = solver.pcs(npcs=npcs, pcscaling=1)
    pc.attrs['long_name'] = source + ' QBO index'
    pc['mode'] = pc.mode + 1
    eof['mode'] = eof.mode + 1
    vf = solver.varianceFraction(npcs)
    errors = solver.northTest(npcs, vfscaled=True)
    ['qbo_' + str(i) for i in pc]
    qbo_ds = xr.Dataset()
    for ar in pc.groupby('mode'):
        qbo_ds['qbo_' + str(ar[0])] = ar[1]
    qbo_ds = qbo_ds.reset_coords(drop=True)
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


def poly_features(X, feature_dim='regressors', degree=2,
                  interaction_only=False, include_bias=False,
                  normalize_poly=False):
    from sklearn.preprocessing import PolynomialFeatures
    import xarray as xr
    from aux_functions_strat import normalize_xr
    sample_dim = [x for x in X.dims if x != feature_dim][0]
    # Vars = ['x' + str(n) for n in range(X.shape[1])]
    # dic = dict(zip(Vars, X[dim_name].values))
    poly = PolynomialFeatures(degree=degree, interaction_only=interaction_only,
                              include_bias=include_bias)
    X_new = poly.fit_transform(X)
    feature_names = [x for x in X[feature_dim].values]
    new_names = poly.get_feature_names(feature_names)
    new_names = [x.replace(' ', '*') for x in new_names]
    X_with_poly_features = xr.DataArray(X_new, dims=[sample_dim, feature_dim])
    X_with_poly_features[sample_dim] = X[sample_dim]
    X_with_poly_features[feature_dim] = new_names
    X_with_poly_features.attrs = X.attrs
    X_with_poly_features.name = 'Polynomial Features'
    if normalize_poly:
        names_to_normalize = list(set(new_names).difference(set(feature_names)))
        Xds = X_with_poly_features.to_dataset(dim=feature_dim)
        for da_name in names_to_normalize:
            Xds[da_name] = normalize_xr(Xds[da_name], norm=1, verbose=False)
        X_with_poly_features = Xds.to_array(dim=feature_dim).T
    return X_with_poly_features


def prepare_regressors(name='Regressors', plot=True, save=False, poly=None,
                       rewrite_file=True, normalize=False):
    """get all the regressors and prepare them save to file.
    replaced prepare_regressors for MLR function"""
    import aux_functions_strat as aux
    import xarray as xr
    import seaborn as sns
    import matplotlib.pyplot as plt
    import os
    path = os.getcwd() + '/'
    # bdc
    # bdc = xr.open_dataarray(path + 'era5_merra_BDC.nc')
    bdc = get_BDC(plot=False, deseason=True)
    bdc.name = 'bdc'
    # t500
    t500 = get_T500(plot=False, deseason=True)
    # t500 = get_T500(False)  # xr.open_dataarray(path + 'era5_merra_T500.nc')
    t500.name = 't500'
    # ENSO
    enso = get_enso_ersst_save_locally()
    enso = enso['ANOM_NINO3.4']
    enso.attrs['long_name'] = enso.name
    enso.name = 'enso'
    # SOLAR
    solar = get_solar_10p7cm_flux_save_locally()
    solar = solar.Adjflux
    # solar = aux.normalize_xr(solar, 1)
    solar.attrs['long_name'] = 'Solar Adjflux 10.7cm'
    solar.name = 'solar'
    # Volcanic forcing
    vol = get_strato_aerosol()
    vol = vol.sad.sel(altitude=20)
    vol = vol.rename({'latitude': 'lat'})
    vol.name = 'vol'
    vol = aux.xr_weighted_mean(vol.sel(lat=slice(-20, 20)))
    # vol = aux.normalize_xr(vol, 4)
    vol.attrs['long_name'] = 'Stratoapheric aerosol density'
    # get the qbo 2 pcs:
    qbo = get_qbo_pcs(2, source='era5', plot=False)  # xr.Dataset()
    qbo_1 = qbo['qbo_1']
    qbo_2 = qbo['qbo_2']
    # get GHG:
    ghg = get_GHG(False)
    ghg.name = 'ghg'
    # get cold point:
    cold = get_cold_point(False)
    # get olr:
    olr = get_OLR(False)
    olr.name = 'olr'
    # get ch4:
    ch4 = get_CH4(False)
    # get wind_shear:
    wind_shear = get_wind_shear(plot=False, source='era5')
    da_list = [qbo_1, qbo_2, solar, enso, vol, t500, bdc, cold, olr, ch4,
               wind_shear]
    ds = xr.Dataset()
    for da in da_list:
        ds[da.name] = da
    # fix vol and ch4
    ds['vol'] = ds['vol'].fillna(1.31)
    ds = ds.reset_coords(drop=True)
    ds['ch4'] = ds['ch4'].fillna(0.019076 + 1.91089)
    if poly is not None:
        da = ds.to_array(dim='regressors').dropna(dim='time').T
        da = poly_features(da, feature_dim='regressors', degree=poly,
                           interaction_only=False, include_bias=False,
                           normalize_poly=False)
        ds = da.to_dataset(dim='regressors')
        name = 'Regressors_d' + str(poly)
    if normalize:
        ds = ds.apply(aux.normalize_xr, norm=1,
                      keep_attrs=True, verbose=False)
    if save:
        if rewrite_file:
            try:
                os.remove(path + name + '.nc')
            except OSError as e:  # if failed, report it back to the user
                print("Error: %s - %s." % (e.filename, e.strerror))
            print('Updating ' + name + '.nc' + ' in ' + path)
        ds.to_netcdf(path + name + '.nc', 'w')
        print(name + ' was saved to ' + path)
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


def prepare_for_MLR(main_name='Regressors', plot=True):
    """get all the regressors and prepare them with the same time for MLR"""
    import aux_functions_strat as aux
    import xarray as xr
    import seaborn as sns
    import matplotlib.pyplot as plt
    import os

    def combine_regressors(reg_data_arrays):
        import xarray as xr
        combine_regressors.counter += 1
        ds = xr.Dataset()
        ds.attrs['group_number'] = combine_regressors.counter
        for da in reg_data_arrays:
            ds[da.name] = da
        if 'vol' in ds.data_vars.keys():
            ds['vol'] = ds['vol'].fillna(1.31)
        # mask vol: drop the time dates:
        # '1991-07-01' to '1996-03-01'
        return ds
    path = os.getcwd() + '/'
    # this means that the files are in dir
    # QBO
    # bdc = get_BDC(plot=False)
    # bdc = bdc.bdc20
    bdc = xr.open_dataarray(path + 'era5_merra_BDC.nc')
    bdc.name = 'bdc'
    # sq = sq.sel(level=50)
    # T500 = get_T500(plot=False)
    # T500 = T500.T20
    t500 = xr.open_dataarray(path + 'era5_merra_T500.nc')
    t500.name = 't500'
    # ENSO
    enso = get_enso_ersst_save_locally(path=path)
    enso = enso['ANOM_NINO3.4']
    enso.attrs['long_name'] = enso.name
    enso.name = 'enso'
    # SOLAR
    solar = get_solar_10p7cm_flux_save_locally(path=path)
    solar = solar.Adjflux
    # solar = aux.normalize_xr(solar, 1)
    solar.attrs['long_name'] = 'Solar Adjflux 10.7cm'
    solar.name = 'solar'
    # Volcanic forcing
    vol = get_strato_aerosol()
    vol = vol.sad.sel(altitude=20)
    vol = vol.rename({'latitude': 'lat'})
    vol.name = 'vol'
    vol = aux.xr_weighted_mean(vol.sel(lat=slice(-20, 20)))
    # vol = aux.normalize_xr(vol, 4)
    vol.attrs['long_name'] = 'Stratoapheric aerosol density'
    # overlap the time:
#    new_time = aux.overlap_time_xr(sq, enso, solar, vol)
#    sq = sq.sel(time=new_time)
#    enso = enso.sel(time=new_time)
#    solar = solar.sel(time=new_time)
#    vol = vol.sel(time=new_time)
    # get the qbo 2 pcs:
    qbo = get_qbo_pcs(2, False)  # xr.Dataset()
    qbo_1 = qbo['qbo_1']
    qbo_2 = qbo['qbo_2']
#    Regressors['qbo'] = sq
    # Regressors['enso'] = enso
    combine_regressors.counter = 0
    reg_list = []
    reg_list.append(combine_regressors([qbo_1, qbo_2, enso, solar, vol]))
    reg_list.append(combine_regressors([qbo_1, qbo_2, solar, vol, t500, bdc]))
    reg_list.append(combine_regressors([qbo_1, qbo_2, enso, solar]))
    reg_list.append(combine_regressors([qbo_1, qbo_2, solar, t500, bdc]))
    for reg_group in reg_list:
        name = main_name + '_' + '%d.nc' % int(reg_group.attrs['group_number'])
        reg_group.to_netcdf(path + name, 'w')
        print(name + ' was saved in dir.')
        if plot:
            df = reg_group.to_dataframe()
            df.plot()
            plt.figure()
            sns.heatmap(df.corr(), annot=True, fmt='.2f')
#    Regressors['solar'] = solar
#    Regressors['vol'] = vol
#    # Regressors = Regressors.drop('level')
#    Regressors['t500'] = T500
#    Regressors['bdc'] = bdc
#    # first take care of vol data so we can extend the time
#    Regressors = Regressors.dropna('time', thresh=5)  # till 1/7/2018
#    Regressors['vol'] = Regressors['vol'].fillna(1.31)  # fill value for no eruption (taken from 1950)
#    Regressors.to_netcdf(path + filename, 'w')
#    print('Regressors were assembeled and save to ' + path + filename)
#    if plot:
#        df = Regressors.to_dataframe()
#        sns.heatmap(df.corr(), annot=True, fmt='.2f')
    return reg_list
    
def get_enso_MEI_save_locally(path='/Users/shlomi/Dropbox/My_backup/Python_projects/Stratosphere_Chaim/',
                              filename='enso_MEI.nc'):
    import os.path
    import io
    import pandas as pd
    import xarray as xr
    import numpy as np
    if os.path.isfile(os.path.join(path, filename)):
        print('NOAA ENSO MEI already d/l and saved!')
        # read it to data array (xarray)
        nino_xr = xr.open_dataset(path + filename)
        # else d/l the file and first read it to df (pandas), then to xarray then save as nc:
    else:
        print('Downloading ENSO MEI data from noaa esrl website...')
        url = 'https://www.esrl.noaa.gov/psd/enso/mei/table.html'
        nino_df = pd.read_html(url)
        # idx = pd.to_datetime(dict(year=nino_df.YR, month=nino_df.MON, day='1'))
        # nino_df = nino_df.set_index(idx)
        # nino_df = nino_df.drop(nino_df.iloc[:, 0:2], axis=1)
        # nino_df.columns = ['NINO1+2', 'ANOM_NINO1+2', 'NINO3', 'ANOM_NINO3',
        #                 'NINO4', 'ANOM_NINO4', 'NINO3.4', 'ANOM_NINO3.4']
        # nino_df = nino_df.rename_axis('time')
        # nino_xr = xr.Dataset(nino_df)
        # nino_xr.to_netcdf(path + filename)
        print('Downloaded NOAA ENSO MEI data and saved it to: ' + filename)
    return nino_df


def get_solar_10p7cm_flux_save_locally(filename='solar_10p7cm.nc'):
    """get the solar flux from Dominion Radio astrophysical Observatory Canada"""
    import ftputil
    import os.path
    import io
    import pandas as pd
    import xarray as xr
    import os
    path = os.getcwd() + '/'
    if os.path.isfile(os.path.join(path, filename)):
        print('Solar flux 10.7cm from DRAO Canada already d/l and saved!')
        # read it to data array (xarray)
        solar_xr = xr.open_dataset(path + filename)
        # else d/l the file and fObsirst read it to df (pandas), then to xarray then save as nc:
    else:
        filename_todl = 'solflux_monthly_average.txt'
        with ftputil.FTPHost('ftp.geolab.nrcan.gc.ca', 'anonymous', '') as ftp_host:
            ftp_host.chdir('/data/solar_flux/monthly_averages/')
            ftp_host.download(filename_todl, path + filename_todl)  
        solar_df = pd.read_csv(path + filename_todl, delim_whitespace=True, skiprows=1)
        print('Downloading solar flux 10.7cm from DRAO Canada website...')
        idx = pd.to_datetime(dict(year=solar_df.Year, month=solar_df.Mon, day='1'))
        solar_df = solar_df.set_index(idx)
        solar_df = solar_df.drop(solar_df.iloc[:, 0:2], axis=1)
        solar_df = solar_df.rename_axis('time')
        solar_xr = xr.Dataset(solar_df)
        solar_xr.attrs['long_name'] = 'Monthly averages of Solar 10.7 cm flux'
        solar_xr.to_netcdf(path + filename)
        print('Downloaded solar flux 10.7cm data and saved it to: ' + filename)
    return solar_xr


def get_strato_aerosol(filename='multiple_input4MIPs_aerosolProperties_CMIP_IACETH-SAGE3lambda-3-0-0_gn_185001_201412.nc'):
    import os.path
    import sys
    import xarray as xr
    import numpy as np
    import pandas as pd
    from datetime import date, timedelta
    if sys.platform == 'linux':
        path = '/home/shlomi/Desktop/DATA/Work Files/Chaim_Stratosphere_Data/'
    elif sys.platform == 'darwin':  # mac os
        path = '/Users/shlomi/Documents/Chaim_Stratosphere_Data/'
    if os.path.isfile(os.path.join(path, filename)):
        aerosol_xr = xr.open_dataset(path + filename, decode_times=False)
        start_date = date(1850, 1, 1)
        days_from = aerosol_xr.time.values.astype('O')
        offset = np.empty(len(days_from), dtype='O')
        for i in range(len(days_from)):
            offset[i] = (start_date + timedelta(days_from[i])).strftime('%Y-%m')
        aerosol_xr['time'] = pd.to_datetime(offset)
        print('Importing ' + path + filename + ' to Xarray')
    else:
        print('File not found...')
        return
    return aerosol_xr


def get_nao_save_locally(path='/Users/shlomi/Dropbox/My_backup/Python_projects/Stratosphere_Chaim/',
                         filename='noaa_nao.nc'):
    import requests
    import os.path
    import io
    import pandas as pd
    import xarray as xr
    import numpy as np
    if os.path.isfile(os.path.join(path, filename)):
        print('Noaa NAO already d/l and saved!')
        # read it to data array (xarray)
        nao_xr = xr.open_dataarray(path + filename)
        # else d/l the file and first read it to df (pandas), then to xarray then save as nc:
    else:
        print('Downloading nao data from noaa ncep website...')
        url = 'http://www.cpc.ncep.noaa.gov/products/precip/CWlink/pna/norm.nao.monthly.b5001.current.ascii'
        s = requests.get(url).content
        nao_df = pd.read_csv(io.StringIO(s.decode('utf-8')), header=None, delim_whitespace=True)
        nao_df.columns = ['YR', 'MON', 'nao']
        idx = pd.to_datetime(dict(year=nao_df.YR, month=nao_df.MON, day='1'))
        nao_df = nao_df.set_index(idx)
        nao_df = nao_df.drop(nao_df.iloc[:, 0:2], axis=1)
        nao_df = nao_df.rename_axis('time')
        nao_df = nao_df.squeeze(axis=1)
        nao_xr = xr.DataArray(nao_df)
        nao_xr.attrs['long_name'] = 'North Atlantic Oscillation'
        nao_xr.name = 'NAO'
        nao_xr.to_netcdf(path + filename)
        print('Downloaded nao data and saved it to: ' + filename)
    return nao_xr


def get_enso_ersst_save_locally(filename='noaa_ersst_nino.nc'):
    import requests
    import os.path
    import io
    import pandas as pd
    import xarray as xr
    import numpy as np
    import os
    path = os.getcwd() + '/'
    if os.path.isfile(os.path.join(path, filename)):
        print('Noaa Ersst El-Nino SO already d/l and saved!')
        # read it to data array (xarray)
        nino_xr = xr.open_dataset(path + filename)
        # else d/l the file and first read it to df (pandas), then to xarray then save as nc:
    else:
        print('Downloading ersst nino data from noaa ncep website...')
        url = 'http://www.cpc.ncep.noaa.gov/data/indices/ersst5.nino.mth.81-10.ascii'
        s = requests.get(url).content
        nino_df = pd.read_csv(io.StringIO(s.decode('utf-8')), delim_whitespace=True)
        idx = pd.to_datetime(dict(year=nino_df.YR, month=nino_df.MON, day='1'))
        nino_df = nino_df.set_index(idx)
        nino_df = nino_df.drop(nino_df.iloc[:, 0:2], axis=1)
        nino_df.columns = ['NINO1+2', 'ANOM_NINO1+2', 'NINO3', 'ANOM_NINO3',
                        'NINO4', 'ANOM_NINO4', 'NINO3.4', 'ANOM_NINO3.4']
        nino_df = nino_df.rename_axis('time')
        nino_xr = xr.Dataset(nino_df)
        nino_xr.to_netcdf(path + filename)
        print('Downloaded ersst_nino data and saved it to: ' + filename)
    return nino_xr


def get_enso_sstoi_save_locally(
        path='/Users/shlomi/Dropbox/My_backup/Python_projects/Stratosphere_Chaim/',
        filename='noaa_sstoi_nino.nc'):
    import requests
    import os.path
    import io
    import pandas as pd
    import xarray as xr
    import numpy as np
    if os.path.isfile(os.path.join(path, filename)):
        print('Noaa Sstoi El-Nino SO already d/l and saved!')
        # read it to data array (xarray)
        nino_xr = xr.open_dataset(path + filename)
        # else d/l the file and first read it to df (pandas), then to xarray then save as nc:
    else:
        print('Downloading sstoi nino data from noaa ncep website...')
        url = 'http://www.cpc.ncep.noaa.gov/data/indices/sstoi.indices'
        s = requests.get(url).content
        nino_df = pd.read_csv(io.StringIO(s.decode('utf-8')), delim_whitespace=True)
        idx = pd.to_datetime(dict(year=nino_df.YR, month=nino_df.MON, day='1'))
        nino_df = nino_df.set_index(idx)
        nino_df = nino_df.drop(nino_df.iloc[:, 0:2], axis=1)
        nino_df.columns = ['NINO1+2', 'ANOM_NINO1+2', 'NINO3', 'ANOM_NINO3',
                        'NINO4', 'ANOM_NINO4', 'NINO3.4', 'ANOM_NINO3.4']
        nino_df = nino_df.rename_axis('time')
        nino_xr = xr.Dataset(nino_df)
        nino_xr.to_netcdf(path + filename)
        print('Downloaded sstoi_nino data and saved it to: ' + filename)
    return nino_xr


def get_solar_250nm_save_locally(filename='nrl2_ssi.nc'):
    import requests
    import os.path
    import io
    import pandas as pd
    import xarray as xr
    import numpy as np
    import os
    from datetime import date, timedelta
    path = os.getcwd() + '/'
    if os.path.isfile(os.path.join(path, filename)):
        print('Solar irridiance 250nm already d/l and saved!')
        # read it to data array (xarray)
        solar_xr = xr.open_dataarray(path + filename)
        # else d/l the file and first read it to df (pandas), then to xarray then save as nc:
    else:
        print('Downloading solar 250nm irridiance data from Lasp Interactive Solar IRridiance Datacenter (LISIRD)...')
        url = 'http://lasp.colorado.edu/lisird/latis/nrl2_ssi_P1M.csv?time,wavelength,irradiance&wavelength=249.5'
        s = requests.get(url).content
        solar_df = pd.read_csv(io.StringIO(s.decode('utf-8')))
        start_date = date(1610, 1, 1)
        days_from = solar_df.iloc[:, 0].values
        offset = np.empty(len(days_from), dtype='O')
        for i in range(len(days_from)):
            offset[i] = (start_date + timedelta(days_from[i])).strftime('%Y-%m')
        solar_df = solar_df.set_index(pd.to_datetime(offset))
        solar_df = solar_df.drop(solar_df.iloc[:, 0:2], axis=1)
        solar_df = solar_df.rename_axis('time').rename_axis('irradiance', axis='columns')
        solar_xr = xr.DataArray(solar_df)
        solar_xr.irradiance.attrs = {'long_name': 'Irradiance',
                                     'units': 'W/m^2/nm'}
        solar_xr.attrs = {'long_name': 'Solar Spectral Irradiance (SSI) at 249.5 nm wavelength from LASP'}
        solar_xr.name = 'Solar UV'
        solar_xr.to_netcdf(path + filename)
        print('Downloaded ssi_250nm data and saved it to: ' + filename)
    return solar_xr


def get_singapore_qbo_save_locally(
        path='/Users/shlomi/Dropbox/My_backup/Python_projects/Stratosphere_Chaim/',
        filename='singapore_qbo.nc'):
    import requests
    import os.path
    import io
    import pandas as pd
    import xarray as xr
    import functools
    """checks the files for the singapore qbo index from Berlin Uni. and
    reads them or downloads them if they are
    missing. output is the xarray and csv backup locally"""
    if os.path.isfile(os.path.join(path, filename)):
        print('singapore QBO already d/l and saved!')
        # read it to data array (xarray)
        sqbo_xr = xr.open_dataarray(path + filename)
        # else d/l the file and first read it to df (pandas), then to xarray then save as nc:
    else:
        print('Downloading singapore data from Berlin university...')
        url = 'http://www.geo.fu-berlin.de/met/ag/strat/produkte/qbo/singapore.dat'
        s = requests.get(url).content
        sing_qbo = sing_qbo = pd.read_csv(io.StringIO(s.decode('utf-8')), skiprows=3,
                                          header=None, delim_whitespace=True,
                                          names=list(range(0, 13)))
        # take out year data
        year = sing_qbo.iloc[0:176:16][0]
        # from 1997 they added another layer (100 hPa) to data hence the irregular indexing:
        year = pd.concat([year, sing_qbo.iloc[177::17][0]], axis=0)
        df_list = []
        # create a list of dataframes to start assembeling data:
        for i in range(len(year)-1):
            df_list.append(pd.DataFrame(data=sing_qbo.iloc[year.index[i]+1:year.index[i+1]]))
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
        df_combined = functools.reduce(lambda df1, df2: df1.join(df2, how='outer'), df_list)
        df_combined = df_combined.rename_axis('level').rename_axis('time', axis='columns')
        df_combined = df_combined.sort_index(ascending=False)
        sqbo_xr = xr.DataArray(df_combined)
        sqbo_xr.level.attrs = {'long_name': 'pressure',
                               'units': 'hPa',
                               'positive': 'down',
                               'axis': 'Z'}
        sqbo_xr.attrs = {'long_name': 'Monthly mean zonal wind components at Singapore (48698), 1N/104E',
                         'units': 'm/s'}
        sqbo_xr.name = 'Singapore QBO'
        sqbo_xr.to_netcdf(path + filename)
        print('Downloaded singapore qbo data and saved it to: ' + filename)
    return sqbo_xr


def plot_time_over_pc_xr(pc, times, norm=5):
    import numpy as np
    import matplotlib.pyplot as plt
    import aux_functions_strat as aux
    min_times = times.time.values.min()
    max_times = times.time.values.max()
    min_pc = pc.time.values.min()
    max_pc = pc.time.values.max()
    max_time = np.array((max_times, max_pc)).min()
    min_time = np.array((min_times, min_pc)).max()
    fig, ax = plt.subplots(figsize=(16, 4))
    pc = aux.normalize_xr(pc, norm)
    times = aux.normalize_xr(times, norm)
    pc.sel(time=slice(min_time, max_time)).plot(c='b', ax=ax)
    times.sel(time=slice(min_time, max_time)).plot(c='r', ax=ax)
    ax.set_title(pc.attrs['long_name'] + ' mode:' + str(pc.mode.values) +
                 ' and ' + times.name)
    ax.set_ylabel('')
    plt.legend(['PC', times.name])
    plt.show()
    return


def get_time_series():
    sq = get_singapore_qbo_save_locally()
    solar = get_solar_250nm_save_locally()
    nino_sstoi = get_enso_sstoi_save_locally()
    nino_ersst = get_enso_ersst_save_locally()
    time_series = []
    time_series.append([sq, solar, nino_sstoi, nino_ersst])
    return time_series
