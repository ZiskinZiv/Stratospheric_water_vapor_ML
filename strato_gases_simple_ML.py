#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 08:54:32 2021
OK so far:
    swoosh h2o: 1994-2019 30S to 30N mean, 82 hpa
    regressors:
    QBO_CDAS = +5 months lag correlated with h2o: 0.508
    Anom_nino3p4 = no lags corr with h2o: -0.167
    LR:
    no CV does R2 of 0.2857
    Cross validate 5 kfolds: mean R2: 0.1786 std R2: 0.245
    SVM:
    CV 5 kfolds: mean R2: 0.418, mean adj_R2: 0.408,
    std R2: 0.047, std adj_R2: 0.0485
    need to plot residuals with best model.
@author: shlomi
"""
from strat_paths import work_chaim
ml_path = work_chaim / 'ML'


# def CV_splitter_for_xarray_time_series(X_da, time_dim='time', grp='year'):
#     groups = X_da.groupby('{}.{}'.format(time_dim, grp)).groups
#     sorted_groups = [value for (key, value) in sorted(groups.items())]
#     cv = [(sorted_groups[i] + sorted_groups[i+1], sorted_groups[i+2])
#           for i in range(len(sorted_groups)-2)]
#     return cv\
def ABS_SHAP(df_shap, df):
    import numpy as np
    import pandas as pd
    import seaborn as sns
    sns.set_theme(style='ticks', font_scale=1.2)
    #import matplotlib as plt
    # Make a copy of the input data
    shap_v = pd.DataFrame(df_shap)
    feature_list = df.columns
    shap_v.columns = feature_list
    df_v = df.copy().reset_index().drop('time', axis=1)

    # Determine the correlation in order to plot with different colors
    corr_list = list()
    for i in feature_list:
        b = np.corrcoef(shap_v[i], df_v[i])[1][0]
        corr_list.append(b)
    corr_df = pd.concat(
        [pd.Series(feature_list), pd.Series(corr_list)], axis=1).fillna(0)
    # Make a data frame. Column 1 is the feature, and Column 2 is the correlation coefficient
    corr_df.columns = ['Predictor', 'Corr']
    corr_df['Sign'] = np.where(corr_df['Corr'] > 0, 'red', 'blue')

    # Plot it
    shap_abs = np.abs(shap_v)
    k = pd.DataFrame(shap_abs.mean()).reset_index()
    k.columns = ['Predictor', 'SHAP_abs']
    k2 = k.merge(corr_df, left_on='Predictor', right_on='Predictor', how='inner')
    k2 = k2.sort_values(by='SHAP_abs', ascending=True)
    colorlist = k2['Sign']
    ax = k2.plot.barh(x='Predictor', y='SHAP_abs',
                      color=colorlist, figsize=(5, 6), legend=False)
    ax.set_xlabel("SHAP Value (Red = Positive Impact)")
    return


def plot_simplified_shap_tree_explainer(rf_model):
    import shap
    X = produce_X(lag={'qbo_cdas': 5})
    y = produce_y(detrend=None)
    X = X.sel(time=slice('1994', '2019'))
    y = y.sel(time=slice('1994', '2019'))
    rf_model.fit(X, y)
    dfX = X.to_dataset('regressor').to_dataframe()
    dfX = dfX.rename(
        {'qbo_cdas': 'QBO', 'anom_nino3p4': 'ENSO', 'co2': r'CO$_2$'}, axis=1)
    ex_rf = shap.Explainer(rf_model)
    shap_values_rf = ex_rf.shap_values(dfX)
    ABS_SHAP(shap_values_rf, dfX)
    return


def plot_Tree_explainer_shap(rf_model):
    import shap
    X = produce_X(lag={'qbo_cdas': 5})
    y = produce_y(detrend=None)
    X = X.sel(time=slice('1994', '2019'))
    y = y.sel(time=slice('1994', '2019'))
    rf_model.fit(X, y)
    dfX = X.to_dataset('regressor').to_dataframe()
    dfX = dfX.rename(
        {'qbo_cdas': 'QBO', 'anom_nino3p4': 'ENSO', 'co2': r'CO$_2$'}, axis=1)
    fi = dict(zip(dfX.columns, rf_model.feature_importances_ * 100))
    print(fi)
    ex_rf = shap.Explainer(rf_model)
    shap_values_rf = ex_rf.shap_values(dfX)
    shap.summary_plot(shap_values_rf, dfX, plot_size=1.1)
    return


def plot_model_prediction_fig_3():
    from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
    from sklearn.linear_model import LinearRegression
    import seaborn as sns
    import matplotlib.pyplot as plt
    X = produce_X()
    X = add_enso2_and_enso_qbo_to_X(X)
    y = produce_y(detrend=None)
    X_test = X.sel(time=slice('1994', '2019'))
    y_test = y.sel(time=slice('1994', '2019'))
    X_train = X.sel(time=slice('2005', '2019'))
    y_train = y.sel(time=slice('2005', '2019'))
    lr = LinearRegression()
    rds = make_results_for_MLR(lr, X_train, y_train, X_test=X_test, y_test=y_test)
    df = rds['predict'].to_dataframe()
    df['y_true'] = y_test.to_dataframe()
    df['resid'] = df['predict'] - df['y_true']
    df = df.rename({'resid': 'Residuals', 'predict': 'MLR', 'y_true': 'SWOOSH'}, axis=1)
    sns.set_theme(style='ticks', font_scale=1.5)
    fig, ax = plt.subplots(2, 1, figsize=(18, 7))
    df[['SWOOSH', 'MLR']].plot(ax=ax[0], color=['tab:purple', 'tab:red'])
    df[['Residuals']].plot(ax=ax[1], color='k', legend=False)
    [x.grid(True) for x in ax]
    [x.set_xlabel('') for x in ax]
    ax[0].set_ylabel(r'H$_{2}$O anomalies [std]')
    ax[1].set_ylabel(r'H$_{2}$O residuals [std]')
    [x.xaxis.set_minor_locator(AutoMinorLocator()) for x in ax]
    [x.xaxis.grid(True, which='minor') for x in ax]
    # legend = ax.legend(prop={'size': 13}, ncol=5, loc='upper left')
    plot_forecast_busts_lines_datetime(ax[0], color='k')
    fig.tight_layout()
    # # get handles and labels of legend:
    # hands, labes = ax.get_legend_handles_labels()
    # colors = [x.get_color() for x in hands]
    # # change the text labels to the colors of the lines:
    # for i, text in enumerate(legend.get_texts()):
    #     text.set_color(colors[i])
    return fig


def plot_beta_coeffs(rds, col_wrap=3, figsize=(13, 6), extent=[-170, 170, -57.5, 57.5], drop_co2=True):
    import cartopy.crs as ccrs
    import seaborn as sns
    import matplotlib.ticker as mticker
    from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
    from palettable.scientific import diverging as divsci
    from strato_figures import remove_regressors_and_set_title
    predict_cmap = divsci.Vik_20.mpl_colormap
    sns.set_theme(style='ticks', font_scale=1.5)
    proj = ccrs.PlateCarree(central_longitude=0)
    plt_kwargs = dict(add_colorbar=False,
                      col_wrap=col_wrap,
                      cmap=predict_cmap, center=0.0, extend='max', vmax=0.6,
                      levels=41, subplot_kws=dict(projection=proj),
                      transform=ccrs.PlateCarree(), figsize=figsize)

    label = r'$\beta$ coefficients'
    gl_list = []
    if drop_co2:
        rds = rds.drop_sel(regressor='co2')
        plt_kwargs.update(extend=None, vmax=None, col_wrap=2)
    fg = rds['params'].plot.contourf(col='regressor', **plt_kwargs)
    cbar_kws = {'label': '', 'format': '%0.2f'}
    cbar_ax = fg.fig.add_axes([0.1, 0.1, .8, .035])  # last num controls width
    fg.add_colorbar(cax=cbar_ax, orientation="horizontal", **cbar_kws)
    for ax in fg.axes.flatten():
        ax.coastlines()
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        gl = ax.gridlines(
            crs=ccrs.PlateCarree(),
            linewidth=1,
            color='black',
            alpha=0.5,
            linestyle='--',
            draw_labels=True)
        gl.xlabels_top = False
        gl.xlabel_style = {'size': 9}
        gl.ylabel_style = {'size': 9}
        gl.xlines = True
        gl.xlocator = mticker.FixedLocator([-180, -120, -60, 0, 60, 120, 180])
        gl.ylocator = mticker.FixedLocator([-45, -30, -15, 0, 15, 30, 45])
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        gl_list.append(gl)
        ax = remove_regressors_and_set_title(ax)
    gl_list[0].ylabels_right = False
    gl_list[1].ylabels_right = False
    gl_list[1].ylabels_left = True
    gl_list[2].ylabels_right = False
    gl_list[3].ylabels_left = True
    gl_list[3].ylabels_right = True
    try:
        gl_list[3].ylabels_right = False
    except IndexError:
        pass
    fg.fig.tight_layout()
    fg.fig.subplots_adjust(top=0.93,
                           bottom=0.2,
                           left=0.05,
                           right=0.979,
                           hspace=0.275,
                           wspace=0.044)

    # fg = rds['params'].plot.contourf(col='regressor', **plt_kwargs)
    # cbar_ax = fg.fig.add_axes([0.1, 0.1, .8, .025])
    # fg.add_colorbar(cax=cbar_ax, orientation="horizontal", label='',
    #                 format='%0.3f')
    # # fg.fig.suptitle(label, fontsize=12, fontweight=750)
    # [ax.coastlines() for ax in fg.axes.flatten()]
    # [ax.gridlines(
    #     crs=ccrs.PlateCarree(),
    #     linewidth=1,
    #     color='black',
    #     alpha=0.5,
    #     linestyle='--',
    #     draw_labels=False) for ax in fg.axes.flatten()]
    # fg.fig.subplots_adjust(bottom=0.2, top=0.9, left=0.05)
    return fg


def plot_r2_map_predictor_sets_with_co2(path=work_chaim, save=True):
    """r2 map (lat-lon) for cdas-plags, enso, ch4"""
    import xarray as xr
    import cartopy.crs as ccrs
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import seaborn as sns
    from strato_figures import remove_regressors_and_set_title
    from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
    from palettable.colorbrewer import sequential as seqbr
    # from palettable.scientific import diverging as divsci
    # from palettable.colorbrewer import diverging as divbr
    from strat_paths import savefig_path

    error_cmap = seqbr.YlGnBu_9.mpl_colormap
    sns.set_theme(style='ticks', font_scale=1.5)

    # rds1 = xr.open_dataset(
    #             path /
    #             'MLR_H2O_latlon_cdas-plags_ch4_enso_2004-2019.nc')
    # rds2 = xr.open_dataset(
    #         path /
    #         'MLR_H2O_latlon_cdas-plags_ch4_enso_bdc_t500_2004-2019.nc')
    # rds3 = xr.open_dataset(
    #         path /
    #         'MLR_H2O_latlon_cdas-plags_ch4_enso_radio_cold_lags6_2004-2019.nc')
    # rds4 = xr.open_dataset(
    #     path /
    #     'MLR_H2O_latlon_cdas-plags_ch4_enso_poly_2_no_qbo^2_no_ch4_extra_2004-2019.nc')
    rds1 = produce_rds_etas(eta=1)
    rds2 = produce_rds_etas(eta=2)
    rds3 = produce_rds_etas(eta=3)
    rds4 = produce_rds_etas(eta=4)
    rds = xr.concat([x['r2'] for x in [rds1, rds2, rds3, rds4]], 'eta')
    rds['eta'] = range(1, 5)
    rds = rds.sortby('eta')
#    fig = plt.figure(figsize=(11, 5))
#    ax = fig.add_subplot(1, 1, 1,
#                         projection=ccrs.PlateCarree(central_longitude=0))
#    ax.coastlines()
    proj = ccrs.PlateCarree(central_longitude=0)
    fg = rds.plot.contourf(col='eta', add_colorbar=False, cmap=error_cmap,
                           vmin=0.0, extend=None, levels=41, col_wrap=2,
                           subplot_kws=dict(projection=proj),
                           transform=ccrs.PlateCarree(), figsize=(13, 6))
#    lons = rds.lon.values[0:int(len(rds.lon.values) / 2)][::2]
#    lons_mirror = abs(lons[::-1])
#    lons = np.concatenate([lons, lons_mirror])
#    lats = rds.lat.values[0:int(len(rds.lat.values) / 2)][::2]
#    lats_mirror = abs(lats[::-1])
#    lats = np.concatenate([lats, lats_mirror])
    # ax.set_xticks(lons, crs=ccrs.PlateCarree())
    # ax.set_yticks(lats, crs=ccrs.PlateCarree())
    # lon_formatter = LongitudeFormatter(zero_direction_label=True)
    # lat_formatter = LatitudeFormatter()
    # ax.xaxis.set_major_formatter(lon_formatter)
    # ax.yaxis.set_major_formatter(lat_formatter)
    cbar_kws = {'label': '', 'format': '%0.2f', 'aspect': 20}
    cbar_ax = fg.fig.add_axes([0.1, 0.1, .8, .025])  # last num controls width
    fg.add_colorbar(cax=cbar_ax, orientation="horizontal", **cbar_kws)
    gl_list = []
    for ax in fg.axes.flatten():
        ax.coastlines()
        gl = ax.gridlines(
            crs=ccrs.PlateCarree(),
            linewidth=1,
            color='black',
            alpha=0.5,
            linestyle='--',
            draw_labels=True)
        gl.xlabels_top = False
        gl.xlabel_style = {'size': 9}
        gl.ylabel_style = {'size': 9}
        gl.xlines = True
        gl.xlocator = mticker.FixedLocator([-180, -120, -60, 0, 60, 120, 180])
        gl.ylocator = mticker.FixedLocator([-45, -30, -15, 0, 15, 30, 45])
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        gl_list.append(gl)
        ax = remove_regressors_and_set_title(ax)
    # gl_list[0].ylabels_right = False
    # gl_list[2].ylabels_left = False
#    try:
#        gl_list[3].ylabels_right = False
#    except IndexError:
#        pass
    fg.fig.tight_layout()
    fg.fig.subplots_adjust(top=0.99,
                           bottom=0.16,
                           left=0.065,
                           right=0.935,
                           hspace=0.0,
                           wspace=0.288)
    print('Caption: ')
    print('The adjusted R^2 for the water vapor anomalies MLR analysis in the 82 hPa level with CH4 ,ENSO, and pressure level lag varied QBO as predictors. This MLR spans from 2004 to 2018')
    filename = 'MLR_H2O_r2_map_82_eta_with_co2.png'
    if save:
        plt.savefig(savefig_path / filename, bbox_inches='tight')
    return fg


def produce_rds_etas(eta=1):
    """ run produce_MLR_2D_for_figs_6_and_7 with regressors:
        eta=1 : co2, anom_nino3p4, qbo_lagged
        eta=2 : co2, anom_nino3p4, qbo_lagged, T500, BDC
        eta=3 : co2, anom_nino3p4, qbo_lagged + 6XCPT_lagged
        eta=4 : co2, anom_nino3p4, qbo_lagged, anom_nino3p4^2, qbo_laggedXanom_nino3p4
        co2 is automatically added"""
    pred = ['qbo_cdas', 'anom_nino3p4']
    if eta == 1:
        print('producing eta {} with {}'.format(eta, pred))
        rds = produce_MLR_2D_for_figs_6_and_7(pred, add_enso2=False)
    elif eta == 2:
        pred = pred + ['era5_bdc70', 'era5_t500']
        print('producing eta {} with {}'.format(eta, pred))
        rds = produce_MLR_2D_for_figs_6_and_7(pred, add_enso2=False)
    elif eta == 3:
        pred = pred + ['radio_cold_no_qbo']
        print('producing eta {} with {}'.format(eta, pred))
        rds = produce_MLR_2D_for_figs_6_and_7(pred, add_enso2=False, reg_shift=['radio_cold_no_qbo', 6])
    elif eta == 4:
        print('producing eta {} with {} and enso^2'.format(eta, pred))
        rds = produce_MLR_2D_for_figs_6_and_7(pred, add_enso2=True)
    return rds


def produce_MLR_2D_for_figs_6_and_7(predictors=['qbo_cdas', 'anom_nino3p4'],
                                    lag={'qbo_cdas': 5}, add_enso2=True,
                                    reg_shift=None):
    from sklearn.linear_model import LinearRegression
    X = produce_X(lag=lag, regressors=predictors, add_co2=True, reg_shift=reg_shift)
    if add_enso2:
        X = add_enso2_and_enso_qbo_to_X(X)
    X = X.sel(time=slice('2005', '2019'))
    y = produce_y(detrend=None, lat_band_mean=None, plevel=82, deseason='std',
                  filename='swoosh_lonlatpress-20deg-5deg.nc', sw_var='combinedanomh2oq')
    y = y.sel(lat=slice(-60, 60))
    y = y.sel(time=X.time)
    lr = LinearRegression()
    rds = make_results_for_MLR(lr, X, y)
    return rds


def make_results_for_MLR(lr, X_train, y_train, X_test=None, y_test=None):
    import xarray as xr
    from sklearn.metrics import r2_score
    if len(y_train.dims) > 1:
        # assume sample dim is time:
        target_dims = [x for x in y_train.dims if x != 'time']
        # infer reg_dim from X:
        reg_dim = [x for x in X_train.dims if x != 'time'][0]
        ys_train = y_train.stack(targets=target_dims)
        # fit the model:
        lr.fit(X_train, ys_train)
        rds = xr.Dataset()
        # produce beta:
        rds['params'] = xr.DataArray(lr.coef_, dims=['targets', reg_dim])
        # produce predict:
        if X_test is not None:
            rds['predict'] = xr.DataArray(lr.predict(X_test), dims=['time', 'targets'])
        else:
            rds['predict'] = xr.DataArray(lr.predict(X_train), dims=['time', 'targets'])
        # produce R^2:
        if y_test is not None:
            ys_test = y_test.stack(targets=target_dims)
            r2 = r2_score(ys_test, rds['predict'], multioutput='raw_values')
        else:
            r2 = r2_score(ys_train, rds['predict'], multioutput='raw_values')
        rds['r2'] = xr.DataArray(r2, dims='targets')
        # dims:
        rds[reg_dim] = X_train[reg_dim]
        rds['time'] = ys_train['time']
        rds['targets'] = ys_train['targets']
        # unstack:
        rds = rds.unstack('targets')
        rds['original'] = y_train
        rds.attrs['sample_dim'] = 'time'
        rds.attrs['feature_dim'] = 'regressor'
    elif len(y_train.dims) == 1:
        reg_dim = [x for x in X_train.dims if x != 'time'][0]
        # fit the model:
        lr.fit(X_train, y_train)
        rds = xr.Dataset()
        # produce beta:
        rds['params'] = xr.DataArray(lr.coef_, dims=[reg_dim])
        # produce predict:
        if X_test is not None:
            rds['predict'] = xr.DataArray(lr.predict(X_test), dims=['time'])
            rds['time'] = y_test['time']
        else:
            rds['predict'] = xr.DataArray(lr.predict(X_train), dims=['time'])
            rds['time'] = y_train['time']
        # produce R^2:
        if y_test is not None:
            r2 = r2_score(y_test, rds['predict'])
        else:
            r2 = r2_score(y_train, rds['predict'])
        rds['r2'] = xr.DataArray(r2)
        # dims:
        rds[reg_dim] = X_train[reg_dim]
        rds['original'] = y_train
        rds.attrs['sample_dim'] = 'time'
        rds.attrs['feature_dim'] = 'regressor'
    return rds


def plot_forecast_busts_lines_datetime(ax, color='r', style='--'):
    import pandas as pd
    dts = ['2010-11', '2011-04', '2015-09', '2016-01', '2016-09', '2017-01']
    dts = [pd.to_datetime(x) for x in dts]
    [ax.axvline(x, c=color, ls=style) for x in dts]
    # three forecast busts:
    # 2010D2011JFM, 2015-OND, 2016-OND
    # ax.axvline('2010-05', c=color, ls=style)
    # ax.axvline('2010-09', c=color, ls=style)
    return ax


def plot_model_predictions(da):
    """ run produce_CV_predictions_for_all_HP_optimized_models first"""
    import seaborn as sns
    import matplotlib.pyplot as plt
    from aux_functions_strat import convert_da_to_long_form_df
    from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
    sns.set_theme(style='ticks', font_scale=1.5)
    df = convert_da_to_long_form_df(da)
    fig, ax = plt.subplots(figsize=(18, 5))
    ax = sns.lineplot(data=df, x='time', y='value', hue='model/obs.',
                      legend=True)
    lw = ax.lines[4].get_linewidth()  # lw of first line
    plt.setp(ax.lines[4], linewidth=2.5)
    ax.grid(True)
    ax.set_xlabel('')
    ax.set_ylabel(r'H$_{2}$O anomalies [std]')
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.xaxis.grid(True, which='minor')
    legend = ax.legend(prop={'size': 13}, ncol=5, loc='upper left')
    plot_forecast_busts_lines_datetime(ax, color='k')
    fig.tight_layout()
    # get handles and labels of legend:
    hands, labes = ax.get_legend_handles_labels()
    colors = [x.get_color() for x in hands]
    # change the text labels to the colors of the lines:
    for i, text in enumerate(legend.get_texts()):
        text.set_color(colors[i])
    return fig


def add_enso2_and_enso_qbo_to_X(X):
    import xarray as xr
    from ML_OOP_stratosphere_gases import poly_features
    feats = [x for x in X.regressor.values if 'qbo' in x or 'nino' in x]
    other_feats = [x for x in X.regressor.values if 'qbo' not in x and 'nino'  not in x]
    X1 = poly_features(X.sel(regressor=feats), feature_dim='regressor')
    X1 = X1.drop_sel(regressor='qbo_cdas^2')
    X = xr.concat([X.sel(regressor=other_feats), X1], 'regressor')
    return X


def produce_CV_predictions_for_all_HP_optimized_models(path=ml_path,
                                                       cv='kfold'):
    import xarray as xr
    X = produce_X()
    y = produce_y()
    X = X.sel(time=slice('1994', '2019'))
    y = y.sel(time=slice('1994', '2019'))
    ml = ML_Classifier_Switcher()
    das = []
    for model_name in ['RF', 'SVM', 'MLP', 'MLR']:
        print('preforming LOO with yearly group for {}.'.format(model_name))
        model = ml.pick_model(model_name)
        if model_name != 'MLR':
            model.set_params(**get_HP_params_from_optimized_model(path=path, model=model_name))
        da = cross_val_predict_da(model, X, y, cv=cv)
        da.name = model_name + ' model'
        das.append(da)
    ds = xr.merge(das)
    ds['SWOOSH'] = y
    da = ds.to_array('model/obs.')
    da.name = 'h2o'
    return da


def cross_val_predict_da(estimator, X, y, cv='kfold'):
    from sklearn.model_selection import LeaveOneGroupOut
    from sklearn.model_selection import KFold
    from sklearn.model_selection import cross_val_predict
    if cv == 'logo':
        logo = LeaveOneGroupOut()
        groups = X['time'].dt.year
        cvr = cross_val_predict(estimator, X, y, groups=groups, cv=logo)
    elif cv == 'kfold':
        kfold = KFold(n_splits=5, shuffle=True, random_state=1)
        cvr = cross_val_predict(estimator, X, y, cv=kfold)
    da_ts = y.copy(data=cvr)
    da_ts.attrs['estimator'] = estimator.__repr__().split('(')[0]
    da_ts.name = da_ts.name + '_' + da_ts.attrs['estimator']
    for key, value in estimator.get_params().items():
        da_ts.attrs[key] = value
    return da_ts


    import seaborn as sns
    import numpy as np
    import matplotlib.pyplot as plt

    def change_width(ax, new_value) :
        for patch in ax.patches :
            current_width = patch.get_width()
            diff = current_width - new_value

            # we change the bar width
            patch.set_width(new_value)

            # we recenter the bar
            patch.set_x(patch.get_x() + diff * .5)

    def show_values_on_bars(axs, fs=12, fw='bold', exclude_bar_num=None):
        import numpy as np
        def _show_on_single_plot(ax, exclude_bar_num=3):
            for i, p in enumerate(ax.patches):
                if i != exclude_bar_num and exclude_bar_num is not None:
                    _x = p.get_x() + p.get_width() / 2
                    _y = p.get_y() + p.get_height()
                    value = '{:.1f}'.format(p.get_height())
                    ax.text(_x, _y, value, ha="right",
                            fontsize=fs, fontweight=fw, zorder=20)

        if isinstance(axs, np.ndarray):
            for idx, ax in np.ndenumerate(axs):
                _show_on_single_plot(ax, exclude_bar_num)
        else:
            _show_on_single_plot(axs, exclude_bar_num)
    sns.set_theme(style='ticks', font_scale=1.5)
    fi_da['regressor'] = ['QBO', 'ENSO']
    df = fi_da.to_dataframe('feature_importance') * 100.0
    df = df.unstack().melt()
    fig, ax = plt.subplots(figsize=(6, 8))
    sns.barplot(data=df, x='regressor', y='value', orient='v', ci='sd',
                ax=ax, hue='regressor', estimator=np.mean, dodge=False)
    ax.set_xlabel('')
    ax.set_ylabel('Feature Importance [%]')
    show_values_on_bars(ax, fs=16, exclude_bar_num=1)
    change_width(ax, 0.31)
    ax.legend(loc='upper right')
    fig.tight_layout()
    return fig


def plot_repeated_kfold_dist(df, model_dict, X, y):
    """run assemble_cvr_dataframe first with strategy=Nonen and add_MLR2"""
    import seaborn as sns
    sns.set_theme(style='ticks', font_scale=1.5)
    in_sample_r2 = {}
    X2 = add_enso2_and_enso_qbo_to_X(X)
    for model_name, model in model_dict.items():
        if model_name == 'MLR2':
            model.fit(X2, y)
            in_sample_r2[model_name] = model.score(X2, y)
        else:
            model.fit(X, y)
            in_sample_r2[model_name] = model.score(X, y)
    print(in_sample_r2)
    df_melted = df.T.melt(var_name='model', value_name=r'R$^2$')
    pal = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:pink']
    fg = sns.displot(data=df_melted, x=r'R$^2$', col="model",
                     kind="hist", col_wrap=2, hue='model', stat='density',
                     kde=True, palette=pal)
    letter = ['a', 'b', 'c', 'd', 'e']
    for i, ax in enumerate(fg.axes):
        label = ax.title.get_text()
        model = label.split('=')[-1].strip()
        title = '({}) model = {}'.format(letter[i], model)
        ax.set_title(title)
        mean = df.T.mean().loc[model]
        std = df.T.std().loc[model]
        median = df.T.median().loc[model]
        in_sample = in_sample_r2[model]
        textstr = '\n'.join((
            r'$\mathrm{mean}=%.2f$' % (mean, ),
            r'$\mathrm{median}=%.2f$' % (median, ),
            r'$\mathrm{std}=%.2f$' % (std, ),
            r'in sample result$=%.2f$' % (in_sample, )))
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
                verticalalignment='top', bbox=props)
    # fg.fig.suptitle('Out of sample testing models comparison')
    # fg.fig.subplots_adjust(top=0.916)
    # fg.fig.tight_layout()
    return fg


def assemble_cvr_dataframe(path=ml_path, score='test_r2', n_splits=5,
                           strategy='LOGO-year', add_MLR2=False):
    import pandas as pd
    rf, rf_model = cross_validate_using_optimized_HP(
        path, model='RF', n_splits=n_splits, strategy=strategy)
    svm, svm_model = cross_validate_using_optimized_HP(
        path, model='SVM', n_splits=n_splits, strategy=strategy)
    mlp, mlp_model = cross_validate_using_optimized_HP(
        path, model='MLP', n_splits=n_splits, strategy=strategy)
    lr, lr_model = cross_validate_using_optimized_HP(
        path, model='MLR', n_splits=n_splits, strategy=strategy)
    lr2, lr2_model = cross_validate_using_optimized_HP(
        path, model='MLR', n_splits=n_splits, strategy=strategy,
        add_MLR2=add_MLR2)
    if add_MLR2:
        df = pd.DataFrame([rf[score], svm[score], mlp[score], lr[score], lr2[score]])
        df.index = ['RF', 'SVM', 'MLP', 'MLR', 'MLR2']
        len_cols = len(df.columns)
        df.columns = ['kfold_{}'.format(x+1) for x in range(len_cols)]
        model_dict = {'RF': rf_model, 'SVM': svm_model,
                      'MLP': mlp_model, 'MLR': lr_model, 'MLR2': lr2_model}
    else:
        df = pd.DataFrame([rf[score], svm[score], mlp[score], lr[score]])
        df.index = ['RF', 'SVM', 'MLP', 'MLR']
        len_cols = len(df.columns)
        df.columns = ['kfold_{}'.format(x+1) for x in range(len_cols)]
        model_dict = {'RF': rf_model, 'SVM': svm_model,
                      'MLP': mlp_model, 'MLR': lr_model}
    return df, model_dict


def cross_validate_using_optimized_HP(path=ml_path, model='SVM', n_splits=5,
                                      n_repeats=20, strategy='LOGO-year',
                                      scorers=['r2', 'r2_adj',
                                               'neg_mean_squared_error',
                                               'explained_variance'],
                                      add_MLR2=False):
    from sklearn.model_selection import cross_validate
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.model_selection import KFold
    from sklearn.model_selection import RepeatedKFold
    from sklearn.model_selection import LeaveOneGroupOut
    from sklearn.model_selection import GroupShuffleSplit
    logo = LeaveOneGroupOut()
    gss = GroupShuffleSplit(n_splits=20, test_size=0.1, random_state=1)
    from sklearn.metrics import make_scorer
    X = produce_X()
    if add_MLR2:
        X = add_enso2_and_enso_qbo_to_X(X)
        print('adding ENSO^2 and ENSO*QBO')
    y = produce_y()
    X = X.sel(time=slice('1994', '2019'))
    y = y.sel(time=slice('1994', '2019'))
    groups = X['time'].dt.year
    scores_dict = {s: s for s in scorers}
    if 'r2_adj' in scorers:
        scores_dict['r2_adj'] = make_scorer(r2_adj_score)
    if 'MLR' not in model:
        hp_params = get_HP_params_from_optimized_model(path, model)
    ml = ML_Classifier_Switcher()
    ml_model = ml.pick_model(model_name=model)
    if 'MLR' not in model:
        ml_model.set_params(**hp_params)
    print(ml_model)
    # cv = TimeSeriesSplit(5)
    # cv = KFold(10, shuffle=True, random_state=1)
    cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats,
                       random_state=1)
    if strategy == 'LOGO-year':
        print('using LeaveOneGroupOut strategy.')
        cvr = cross_validate(ml_model, X, y, scoring=scores_dict, cv=logo,
                             groups=groups)
    elif strategy == 'GSS-year':
        print('using GroupShuffleSplit strategy.')
        cvr = cross_validate(ml_model, X, y, scoring=scores_dict, cv=gss,
                             groups=groups)
    else:
        cvr = cross_validate(ml_model, X, y, scoring=scores_dict, cv=cv)
    return cvr, ml_model


def manual_cross_validation_for_RF_feature_importances(rf_model, n_splits=5, n_repeats=20, scorers=['r2', 'r2_adj',
                                                                                                    'neg_mean_squared_error',
                                                                                                    'explained_variance']):
    from sklearn.model_selection import KFold
    import xarray as xr
    import numpy as np
    from sklearn.model_selection import RepeatedKFold
    from sklearn.metrics import make_scorer
    scores_dict = {s: s for s in scorers}
    if 'r2_adj' in scorers:
        scores_dict['r2_adj'] = make_scorer(r2_adj_score)
    print(rf_model)
    X = produce_X()
    y = produce_y()
    X = X.sel(time=slice('1994', '2019'))
    y = y.sel(time=slice('1994', '2019'))
    # cv = TimeSeriesSplit(5)
    # cv = KFold(10, shuffle=True, random_state=1)
    cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats,
                       random_state=1)
    fis = []
    for train_index, test_index in cv.split(X):
        # print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        rf_model.fit(X_train, y_train)
        fis.append(rf_model.feature_importances_)
    fi = xr.DataArray(fis, dims=['repeats', 'regressor'])
    fi['repeats'] = np.arange(1, len(fis)+1)
    fi['regressor'] = X['regressor']
    return fi


def get_HP_params_from_optimized_model(path=ml_path, model='SVM'):
    import joblib
    from aux_functions_strat import path_glob
    files = path_glob(path, 'GRSRCHCV_*.pkl')
    file = [x for x in files if model in x.as_posix()][0]
    gr = joblib.load(file)
    df = read_one_gridsearchcv_object(gr)
    return df.iloc[0][:-2].to_dict()


def produce_X(regressors=['qbo_cdas', 'anom_nino3p4'],
              lag={'qbo_cdas': 5}, add_co2=True, standertize=True,
              reg_shift=None):
    """reg_shift is dict = {regressor: n} where n is the number of times to
    shift backwards one month"""
    from make_regressors import load_all_regressors
    from ML_OOP_stratosphere_gases import regressor_shift
    import xarray as xr
    ds = load_all_regressors()
    ds = ds[regressors].dropna('time')
    if lag is not None:
        for key, value in lag.items():
            print(key, value)
            ds[key] = ds[key].shift(time=value)
    if standertize:
        ds = (ds - ds.mean('time')) / ds.std('time')
    if add_co2:
        ds['co2'] = produce_co2_trend()
    if reg_shift is not None:
        dss = regressor_shift(ds[reg_shift[0]].dropna('time'), shifts=[1,reg_shift[-1]])
        ds = xr.merge([ds, dss])
    X = ds.dropna('time').to_array('regressor')
    X = X.transpose('time', 'regressor')
    return X


def produce_y(path=work_chaim, detrend='lowess',
              sw_var='combinedeqfillanomfillh2oq', filename='swoosh_latpress-2.5deg.nc',
              lat_band_mean=[-5, 5], plevel=82, deseason='mean', standertize=True):
    import xarray as xr
    from aux_functions_strat import lat_mean
    from aux_functions_strat import detrend_ts
    from aux_functions_strat import anomalize_xr
    file = path / filename
    da = xr.open_dataset(file)[sw_var]
    if plevel is not None:
        da = da.sel(level=plevel, method='nearest')
    if lat_band_mean is not None:
        da = lat_mean(da)
    if detrend is not None:
        if detrend == 'lowess':
            da = detrend_ts(da)
    if deseason is not None:
        da = anomalize_xr(da, freq='MS', units=deseason, time_dim='time')
    if standertize is not None:
        da = (da - da.mean('time')) / da.std('time')
    y = da
    return y


def produce_co2_trend(standertize=True):
    from make_regressors import load_all_regressors
    from aux_functions_strat import loess_curve
    ds = load_all_regressors()
    co2 = ds['co2'].dropna('time')
    trend = loess_curve(co2, plot=False)
    if standertize:
        co2 = (trend['mean']-trend['mean'].mean('time')) / \
            trend['mean'].std('time')
        return co2
    else:
        return trend['mean']


def r2_adj_score(y_true, y_pred, **kwargs):
    from sklearn.metrics import r2_score
    r2 = r2_score(y_true, y_pred)
    n = len(y_true)
    if 'p' in kwargs:
        p = kwargs['p']
    else:
        p = 2
    r2_adj = 1.0 - (1.0 - r2) * (n - 1.0) / (n - p)
    # r2_adj = 1-(1-r2)*(n-1)/(n-p-1)
    return r2_adj


def single_cross_validation(X_val, y_val, model_name='SVM',
                            n_splits=5, scorers=['r2', 'r2_adj',
                                                 'neg_mean_squared_error',
                                                 'explained_variance'],
                            seed=42, savepath=None, verbose=0,
                            param_grid='dense', n_jobs=-1):
    # from sklearn.model_selection import cross_validate
    from sklearn.model_selection import StratifiedKFold
    from sklearn.model_selection import KFold
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.model_selection import GridSearchCV
    # from sklearn.model_selection import train_test_split
    from sklearn.metrics import make_scorer
    # from string import digits
    # import numpy as np
    # import xarray as xr
    scores_dict = {s: s for s in scorers}
    if 'r2_adj' in scorers:
        scores_dict['r2_adj'] = make_scorer(r2_adj_score)

    X = X_val.dropna('time').sel(time=y_val['time'])
    y = y_val

    # if param_grid == 'light':
    #     print(np.unique(X.feature.values))

    # configure the cross-validation procedure
    # cv = TimeSeriesSplit(n_splits=n_splits)
    cv = KFold(n_splits=n_splits, random_state=seed, shuffle=True)
    # print('CV TimeSeriesKfolds of {}.'.format(n_splits))
    print('CV KFold of {}.'.format(n_splits))
    # define the model and search space:

    ml = ML_Classifier_Switcher()
    print('param grid group is set to {}.'.format(param_grid))
    # if outer_split == '1-1':
    #     cv_type = 'holdout'
    #     print('holdout cv is selected.')
    # else:
    #     cv_type = 'nested'
    #     print('nested cv {} out of {}.'.format(
    #         outer_split.split('-')[0], outer_split.split('-')[1]))
    sk_model = ml.pick_model(model_name, pgrid=param_grid)
    search_space = ml.param_grid
    # define search
    gr_search = GridSearchCV(estimator=sk_model, param_grid=search_space,
                             cv=cv, n_jobs=n_jobs,
                             scoring=scores_dict,
                             verbose=verbose,
                             refit=False, return_train_score=True)

    gr_search.fit(X, y)
    features = [x for x in X['regressor'].values]
    if savepath is not None:
        filename = 'GRSRCHCV_{}_{}_{}_{}_{}_{}.pkl'.format(model_name, '+'.join(features), '+'.join(
            scorers), n_splits,
            param_grid, seed)
        save_gridsearchcv_object(gr_search, savepath, filename)
    return gr_search


def save_gridsearchcv_object(GridSearchCV, savepath, filename):
    import joblib
    print('{} was saved to {}'.format(filename, savepath))
    joblib.dump(GridSearchCV, savepath / filename)
    return


def load_one_gridsearchcv_object(path=ml_path, model_name='SVM', verbose=True):
    """load one gridsearchcv obj with model_name and features and run read_one_gridsearchcv_object"""
    from aux_functions_strat import path_glob
    import joblib
    # first filter for model name:
    if verbose:
        print('loading GridsearchCVs results for {} model'.format(model_name))
    model_files = path_glob(path, 'GRSRCHCV_*.pkl')
    model_files = [x for x in model_files if model_name in x.as_posix()]
    # now select features:
    # if verbose:
    #     print('loading GridsearchCVs results with {} features'.format(features))
    # model_features = [x.as_posix().split('/')[-1].split('_')[3] for x in model_files]
    # feat_ind = get_feature_set_from_list(model_features, features)
    # also get the test ratio and seed number:
    # if len(feat_ind) > 1:
    #     if verbose:
    #         print('found {} GR objects.'.format(len(feat_ind)))
    #     files = sorted([model_files[x] for x in feat_ind])
    #     outer_splits = [x.as_posix().split('/')[-1].split('.')[0].split('_')[-3] for x in files]
    #     grs = [joblib.load(x) for x in files]
    #     best_dfs = [read_one_gridsearchcv_object(x) for x in grs]
    #     di = dict(zip(outer_splits, best_dfs))
    #     return di
    # else:
        # file = model_files[feat_ind]
        # seed = file.as_posix().split('/')[-1].split('.')[0].split('_')[-1]
        # outer_splits = file.as_posix().split('/')[-1].split('.')[0].split('_')[-3]
    # load and produce best_df:
    gr = joblib.load(model_files[0])
    best_df = read_one_gridsearchcv_object(gr)
    return best_df


def read_one_gridsearchcv_object(gr):
    """read one gridsearchcv multimetric object and
    get the best params, best mean/std scores"""
    import pandas as pd
    # first get all the scorers used:
    scorers = [x for x in gr.scorer_.keys()]
    # now loop over the scorers:
    best_params = []
    best_mean_scores = []
    best_std_scores = []
    for scorer in scorers:
        df_mean = pd.concat([pd.DataFrame(gr.cv_results_["params"]), pd.DataFrame(
            gr.cv_results_["mean_test_{}".format(scorer)], columns=[scorer])], axis=1)
        df_std = pd.concat([pd.DataFrame(gr.cv_results_["params"]), pd.DataFrame(
            gr.cv_results_["std_test_{}".format(scorer)], columns=[scorer])], axis=1)
        # best index = highest score:
        best_ind = df_mean[scorer].idxmax()
        best_mean_scores.append(df_mean.iloc[best_ind][scorer])
        best_std_scores.append(df_std.iloc[best_ind][scorer])
        best_params.append(df_mean.iloc[best_ind].to_frame().T.iloc[:, :-1])
    best_df = pd.concat(best_params)
    best_df['mean_score'] = best_mean_scores
    best_df['std_score'] = best_std_scores
    best_df.index = scorers
    return best_df


def order_of_mag(minimal=-5, maximal=1):
    import numpy as np
    return [10**float(x) for x in np.arange(minimal, maximal + 1)]


class ML_Classifier_Switcher(object):

    def pick_model(self, model_name, pgrid='normal'):
        """Dispatch method"""
        # from sklearn.model_selection import GridSearchCV
        self.param_grid = None
        method_name = str(model_name)
        # Get the method from 'self'. Default to a lambda.
        method = getattr(self, method_name, lambda: "Invalid ML Model")
#        if gridsearch:
#            return(GridSearchCV(method(), self.param_grid, n_jobs=-1,
#                                return_train_score=True))
#        else:
        # Call the method as we return it
        # whether to select lighter param grid, e.g., for testing purposes.
        self.pgrid = pgrid
        return method()

    def SVM(self):
        from sklearn.svm import SVR
        import numpy as np
        if self.pgrid == 'light':
            self.param_grid = {'kernel': ['poly'],
                               'C': [0.1],
                               'gamma': [0.0001],
                               'degree': [1, 2],
                               'coef0': [1, 4]}
        # elif self.pgrid == 'normal':
        #     self.param_grid = {'kernel': ['rbf', 'sigmoid', 'linear', 'poly'],
        #                        'C': order_of_mag(-1, 2),
        #                        'gamma': order_of_mag(-5, 0),
        #                        'degree': [1, 2, 3, 4, 5],
        #                        'coef0': [0, 1, 2, 3, 4]}
        elif self.pgrid == 'dense':
            # self.param_grid = {'kernel': ['rbf', 'sigmoid', 'linear', 'poly'],
            #                    'C': np.logspace(-2, 2, 10), # order_of_mag(-2, 2),
            #                    'gamma': np.logspace(-5, 1, 14), # order_of_mag(-5, 0),
            #                    'degree': [1, 2, 3, 4, 5],
            #                    'coef0': [0, 1, 2, 3, 4]}
            self.param_grid = {'kernel': ['rbf', 'sigmoid', 'linear'],
                               'C': np.logspace(-2, 2, 10), # order_of_mag(-2, 2),
                                'gamma': np.logspace(-5, 1, 14)}#, # order_of_mag(-5, 0),
                               # 'degree': [1, 2, 3, 4, 5],
                               # 'coef0': [0, 1, 2, 3, 4]}
        return SVR()

    def MLP(self):
        import numpy as np
        from sklearn.neural_network import MLPRegressor
        if self.pgrid == 'light':
            self.param_grid = {
                'activation': [
                    'identity',
                    'relu'],
                'hidden_layer_sizes': [(50, 50, 50), (50, 100, 50)]}
        # elif self.pgrid == 'normal':
        #     self.param_grid = {'alpha': order_of_mag(-5, 1),
        #                        'activation': ['identity', 'logistic', 'tanh', 'relu'],
        #                        'hidden_layer_sizes': [(50, 50, 50), (50, 100, 50), (100,)],
        #                        'learning_rate': ['constant', 'adaptive'],
        #                        'solver': ['adam', 'lbfgs', 'sgd']}
        elif self.pgrid == 'dense':
            self.param_grid = {'alpha': np.logspace(-5, 1, 7),
                               'activation': ['identity', 'logistic', 'tanh', 'relu'],
                               'hidden_layer_sizes': [(10, 10, 10), (10, 20, 10), (10,), (5,), (1,)],
                               'learning_rate': ['constant', 'adaptive'],
                               'solver': ['adam', 'lbfgs', 'sgd']}
            #(1,),(2,),(3,),(4,),(5,),(6,),(7,),(8,),(9,),(10,),(11,), (12,),(13,),(14,),(15,),(16,),(17,),(18,),(19,),(20,),(21,)
        return MLPRegressor(random_state=42, max_iter=500, learning_rate_init=0.1)

    def RF(self):
        from sklearn.ensemble import RandomForestRegressor
        # import numpy as np
        if self.pgrid == 'light':
            self.param_grid = {'max_features': ['auto', 'sqrt']}
        elif self.pgrid == 'normal':
            self.param_grid = {'max_depth': [5, 10, 25, 50, 100],
                               'max_features': ['auto', 'sqrt'],
                               'min_samples_leaf': [1, 2, 5, 10],
                               'min_samples_split': [2, 5, 15, 50],
                               'n_estimators': [100, 300, 700, 1200]
                               }
        elif self.pgrid == 'dense':
            self.param_grid = {'max_depth': [5, 10, 25, 50, 100, 150, 250],
                               'max_features': ['auto', 'sqrt'],
                               'min_samples_leaf': [1, 2, 5, 10, 15, 25],
                               'min_samples_split': [2, 5, 15, 30, 50, 70, 100],
                               'n_estimators': [100, 200, 300, 500, 700, 1000, 1300, 1500]
                               }
        return RandomForestRegressor(random_state=42, n_jobs=-1)

    def MLR(self):
        from sklearn.linear_model import LinearRegression
        return LinearRegression(n_jobs=-1)
