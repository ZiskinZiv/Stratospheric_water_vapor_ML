#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 10:26:13 2020

@author: shlomi
"""

from strat_paths import work_chaim
from matplotlib import rcParams
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import matplotlib.dates as mdates
import seaborn as sns
import matplotlib.ticker as mticker
from palettable.scientific import sequential as seqsci
from palettable.colorbrewer import sequential as seqbr
from palettable.scientific import diverging as divsci
from palettable.colorbrewer import diverging as divbr
from matplotlib.colors import ListedColormap
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from pathlib import Path
from strat_paths import savefig_path

error_cmap = seqsci.Nuuk_11.mpl_colormap
error_cmap = seqbr.YlGnBu_9.mpl_colormap
# predict_cmap = ListedColormap(divbr.BrBG_11.mpl_colors)
predict_cmap = divsci.Vik_20.mpl_colormap
# predict_cmap = ListedColormap(divsci.Vik_20.mpl_colors)
rc = {
    'font.family': 'serif',
    'xtick.labelsize': 'medium',
    'ytick.labelsize': 'medium'}
for key, val in rc.items():
    rcParams[key] = val
sns.set(rc=rc, style='ticks')
fields_dict = {'r2_adj': r'Adjusted R$^2$', 'params': r'$\beta$ coeffs'}

def add_horizontal_colorbar(fg_obj, width=0.025, cbar_kwargs_dict=None):
    # add option for just figure object, now, accepts facetgrid object only
    cbar_kws = {'label': '', 'format': '%0.2f'}
    if cbar_kwargs_dict is not None:
        cbar_kws.update(cbar_kwargs_dict)
    cbar_ax = fg_obj.fig.add_axes([0.1, 0.1, .8, width])  # last num controls width
    fg_obj.add_colorbar(cax=cbar_ax, orientation="horizontal", **cbar_kws)
    return fg_obj

def parse_quantile(rds, quan):
    vals = rds.quantile(quan)
    vals = [abs(x) for x in vals]
    if vals[0] < vals[1]:
        quan_kws = {'vmin': -vals[0]}
    else:
        quan_kws = {'vmax': vals[1]}
    return quan_kws


def plot_forecast_busts_lines(ax, color='r', style='--'):
    # three forecast busts:
    # 2010D2011JFM, 2015-OND, 2016-OND
    # ax.axvline('2010-05', c=color, ls=style)
    # ax.axvline('2010-09', c=color, ls=style)
    ax.axvline('2010-11', c=color, ls=style)
    # ax.axvline('2010-12', c=color, ls=style)
    ax.axvline('2011-04', c=color, ls=style)
    ax.axvline('2015-09', c=color, ls=style)
    ax.axvline('2016-01', c=color, ls=style)
    ax.axvline('2016-09', c=color, ls=style)
    ax.axvline('2017-01', c=color, ls=style)
    return ax


def remove_regressors_and_set_title(ax, set_title_only=None):
    short_titles = {'qbo_cdas': 'QBO',
                    'anom_nino3p4': 'ENSO',
                    'ch4': 'CH4',
                    'era5_bdc': 'BDC',
                    'era5_t500': 'T at 500hPa'}
    title = ax.get_title()
    title = title.split('=')[-1].strip(' ')
    if set_title_only is not None:
        title = short_titles.get(set_title_only)
    else:
        title = short_titles.get(title)
    ax.set_title(title)
    return ax


def change_ticks_lat(fig, ax, draw=True, which_axes='x'):
    if draw:
        fig.canvas.draw()
    if which_axes == 'x':
        labels = [item.get_text() for item in ax.get_xticklabels()]
    elif which_axes == 'y':
        labels = [item.get_text() for item in ax.get_yticklabels()]
    labels = [x.replace('−', '-') for x in labels]
    labels = [int(x) for x in labels]
    xlabels = []
    for label in labels:
        if label < 0:
            xlabel = r'{}$\degree$S'.format(str(label).split('-')[-1])
        elif label > 0:
            xlabel = r'{}$\degree$N'.format(label)
        elif label == 0:
            xlabel = r'0$\degree$'
        xlabels.append(xlabel)
    if which_axes == 'x':
        ax.set_xticklabels(xlabels)
    elif which_axes == 'y':
        ax.set_yticklabels(xlabels)
    return ax
    # ax.set_xticklabels(xlabels)


def change_ticks_lon(fig, ax, draw=True, which_axes='x'):
    if draw:
        fig.canvas.draw()
    if which_axes == 'x':
        labels = [item.get_text() for item in ax.get_xticklabels()]
    elif which_axes == 'y':
        labels = [item.get_text() for item in ax.get_yticklabels()]
    labels = [x.replace('−', '-') for x in labels]
    labels = [int(x) for x in labels]
    xlabels = []
    for label in labels:
        if label < 0:
            xlabel = r'{}$\degree$W'.format(str(label).split('-')[-1])
        elif label > 0:
            xlabel = r'{}$\degree$E'.format(label)
        elif label == 0:
            xlabel = r'$\degree$0'
        xlabels.append(xlabel)
    if which_axes == 'x':
        ax.set_xticklabels(xlabels)
    elif which_axes == 'y':
        ax.set_yticklabels(xlabels)
    return ax


def change_xticks_years(ax, start=1984, end=2018):
    import pandas as pd
    import numpy as np
    years_fmt = mdates.DateFormatter('%Y')
    years = np.arange(start, end, 2)
    years = [pd.to_datetime(str(x)).strftime('%Y') for x in years]
    ax.set_xticks(years)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(years_fmt)
    return ax


def plot_figure_1(path=work_chaim, regressors=['qbo_cdas']):
    from ML_OOP_stratosphere_gases import run_ML
    # sns.set_style('ticks', rc=rc)
    cbar_kws = {'label': '', 'format': '%0.2f'}
    if len(regressors) == 1:
        rds = run_ML(time_period=['1984', '2018'], regressors=regressors,
                     special_run={'optimize_reg_shift': [0, 12]},
                     area_mean=True, lat_slice=[-20, 20])
        fg = rds.r2_adj.T.plot.pcolormesh(yscale='log', yincrease=False,
                                          levels=21, col='reg_shifted',
                                          cmap=error_cmap, vmin=0.0,
                                          extend='both', figsize=(7, 7),
                                          cbar_kwargs=cbar_kws)
        ax = fg.axes[0][0]
        ax.yaxis.set_major_formatter(ScalarFormatter())

        rds.isel(reg_shifted=0).level_month_shift.plot.line('r.-', y='level',
                                                            yincrease=False,
                                                            ax=ax)
        ax.set_xlabel('lag [months]')
        ax.set_title('')
        fg.fig.tight_layout()
        fg.fig.subplots_adjust(right=0.8)
        print('Caption:')
        print('The adjusted R^2 for the QBO index from CDAS as a function of pressure level and month lag. The solid line and dots represent the maximum R^2 for each pressure level.')
        filename = 'r2_{}_shift_optimize.png'.format(regressors[0])
    else:
        rds = run_ML(time_period=['1984', '2018'], regressors=regressors,
                     special_run={'optimize_reg_shift': [0, 12]},
                     area_mean=True, lat_slice=[-20, 20])
        fg = rds.r2_adj.T.plot.pcolormesh(yscale='log', yincrease=False,
                                          levels=21, col='reg_shifted',
                                          cmap=error_cmap, vmin=0.0,
                                          extend='both', figsize=(13, 4),
                                          add_colorbar=False)
        cbar_kws = {'label': '', 'format': '%0.2f'}
        cbar_ax = fg.fig.add_axes([0.1, 0.1, .8, .025])
        fg.add_colorbar(cax=cbar_ax, orientation="horizontal", **cbar_kws)
        for n_regs in range(len(fg.axes[0])):
            rds.isel(reg_shifted=n_regs).level_month_shift.plot.line('r.-',
                                                                     y='level',
                                                                     yincrease=False,
                                                                     ax=fg.axes[0][n_regs])
        for ax in fg.axes.flatten():
            ax.set_xlabel('lag [months]')
            ax = remove_regressors_and_set_title(ax)
            ax.yaxis.set_major_formatter(ScalarFormatter())
            ax.set_ylabel('')
        fg.axes[0][0].set_ylabel('Pressure [hPa]')
        fg.fig.tight_layout()
        fg.fig.subplots_adjust(left=0.07, bottom=0.27)
        print('Caption:')
        print('The adjusted R^2 for the QBO, BDC and T500 predictors as a function of pressure level and month lag. The solid line and dots represent the maximum R^2 for each pressure level.')
        filename = 'r2_{}_shift_optimize.png'.format(
            '_'.join([x for x in regressors]))
    plt.savefig(savefig_path / filename, bbox_inches='tight')
    return fg


def plot_figure_2(path=work_chaim, robust=False):
    from ML_OOP_stratosphere_gases import plot_like_results
    import xarray as xr
    rds = xr.open_dataset(
        path /
        'MLR_H2O_latpress_cdas-plags_ch4_enso_1984-2018.nc')
    fg = plot_like_results(rds, plot_key='predict_level-time', lat=None,
                           cmap=predict_cmap, robust=robust, extend=None)
    top_ax = fg.axes[0][0]
    mid_ax = fg.axes[1][0]
    bottom_ax = fg.axes[-1][0]
    # remove time from xlabel:
    bottom_ax.set_xlabel('')
    # new ticks:
    ax = change_xticks_years(bottom_ax, start=1985, end=2018)
    fg.fig.tight_layout()
    fg.fig.subplots_adjust(left=0.06, bottom=0.185)
    top_ax.set_title(
        'Area-averaged (weighted by cosine of latitudes 60S to 60N) combined H2O anomaly')
    mid_ax.set_title('MLR reconstruction')
    bottom_ax.set_title('Residuals')
    print('Caption: ')
    print('Stratospheric water vapor anomalies and their MLR reconstruction and residuals, spanning from 1984 to 2018 and using CH4, ENSO and pressure level lag varied QBO as predictors')
    filename = 'MLR_H2O_predict_level-time_cdas-plags_ch4_enso.png'
    plt.savefig(savefig_path / filename, bbox_inches='tight')
    return fg


def plot_figure_3(path=work_chaim, robust=False):
    from ML_OOP_stratosphere_gases import plot_like_results
    import xarray as xr
    rds = xr.open_dataset(
        path /
        'MLR_H2O_latpress_cdas-plags_ch4_enso_1984-2018.nc')
    fg = plot_like_results(rds, plot_key='r2_level-lat', cmap=error_cmap,
                           robust=robust, extend=None)
    fg.ax.set_title('')
    change_ticks_lat(fg.ax.figure, fg.ax)
    fg.ax.figure.tight_layout()
    fg.ax.figure.subplots_adjust(left=0.15, right=0.95)
    fg.ax.set_xlabel('')
    print('Caption: ')
    print('The adjusted R^2 for the water vapor MLR analysis(1984-2018) with CH4, ENSO and pressure level lag varied QBO as predictors')
    filename = 'MLR_H2O_r2_level-lat_cdas-plags_ch4_enso.png'
    plt.savefig(savefig_path / filename, bbox_inches='tight')
    return fg


def plot_figure_4(path=work_chaim, robust=False):
    from ML_OOP_stratosphere_gases import plot_like_results
    import xarray as xr
    rds = xr.open_dataset(
        path /
        'MLR_H2O_latpress_cdas-plags_ch4_enso_1984-2018.nc')
    fg = plot_like_results(rds, plot_key='params_level-lat', cmap=predict_cmap,
                           figsize=(10, 5), robust=robust, extend=None)
    fg.fig.suptitle('')
    fg.fig.canvas.draw()
    for ax in fg.axes.flatten():
        change_ticks_lat(fg.fig, ax, draw=False)
        ax.set_xlabel('')
        ax = remove_regressors_and_set_title(ax)
    fg.fig.tight_layout()
    fg.fig.subplots_adjust(left=0.1, bottom=0.25)
    print('Caption: ')
    print('The beta coefficiants for the water vapor MLR analysis(1984-2018) with CH4, ENSO and pressure level lag varied QBO as predictors')
    filename = 'MLR_H2O_params_level-lat_cdas-plags_ch4_enso.png'
    plt.savefig(savefig_path / filename, bbox_inches='tight')
    return fg


def plot_figure_5(path=work_chaim, quan=[0.0, 1.0]):
    from ML_OOP_stratosphere_gases import plot_like_results
    import xarray as xr
    rds = xr.open_dataset(
        path /
        'MLR_H2O_latlon_cdas-plags_ch4_enso_2004-2018.nc')
    quan_kws = parse_quantile(rds.resid, quan)
    fg = plot_like_results(rds, plot_key='predict_lat-time', level=82,
                           cmap=predict_cmap, **quan_kws, extend=None)
    top_ax = fg.axes[0][0]
    mid_ax = fg.axes[1][0]
    bottom_ax = fg.axes[-1][0]
    # remove time from xlabel:
    bottom_ax.set_xlabel('')
    # new ticks:
    ax = change_xticks_years(bottom_ax, start=2005, end=2018)
    fg.fig.tight_layout()
    fg.fig.subplots_adjust(left=0.06, bottom=0.185)
    top_ax.set_title(
        'Area-averaged (from 180W to 180E longitudes) combined H2O anomaly for the 82 hPa pressure level')
    mid_ax.set_title('MLR reconstruction')
    bottom_ax.set_title('Residuals')
    fg.fig.canvas.draw()
    for axes in fg.axes.flatten():
        ax = change_ticks_lat(fg.fig, axes, which_axes='y', draw=False)
        ax.set_ylabel('')
        ax = plot_forecast_busts_lines(ax, color='k')
    print('Caption: ')
    print('The zonal mean water vapor anomalies for the 82 hPa level and their MLR reconstruction and residuals, spanning from 2004 to 2018. This MLR analysis was carried out with CH4 ,ENSO and pressure level lag varied QBO as predictors. Note the four forecast "busts": 2010-JJA, 2011-JFM,2015-OND and 2016-OND')
    filename = 'MLR_H2O_predict_lat-time_82_cdas-plags_ch4_enso.png'
    plt.savefig(savefig_path / filename, bbox_inches='tight')
    return fg


def plot_figure_6(path=work_chaim, robust=False):
    from ML_OOP_stratosphere_gases import plot_like_results
    import xarray as xr
    rds = xr.open_dataset(
        path /
        'MLR_H2O_latlon_cdas-plags_ch4_enso_2004-2018.nc')
    fg = plot_like_results(rds, plot_key='predict_lon-time', level=82,
                           cmap=predict_cmap, robust=robust, extend=None)
    top_ax = fg.axes[0][0]
    mid_ax = fg.axes[1][0]
    bottom_ax = fg.axes[-1][0]
    # remove time from xlabel:
    bottom_ax.set_xlabel('')
    # new ticks:
    ax = change_xticks_years(bottom_ax, start=2005, end=2018)
    fg.fig.tight_layout()
    fg.fig.subplots_adjust(left=0.06, bottom=0.185)
    top_ax.set_title(
        'Area-averaged (weighted by cosine of latitudes 60S to 60N) combined H2O anomaly for the 82 hPa pressure level')
    mid_ax.set_title('MLR reconstruction')
    bottom_ax.set_title('Residuals')
    fg.fig.canvas.draw()
    for axes in fg.axes.flatten():
        ax = change_ticks_lon(fg.fig, axes, which_axes='y', draw=False)
        ax.set_ylabel('')
        ax = plot_forecast_busts_lines(ax, color='k')
    print('Caption: ')
    print('The meridional mean water vapor anomalies for the 82 hPa level and their MLR reconstruction and residuals, spanning from 2004 to 2018. This MLR analysis was carried out with CH4 ,ENSO and pressure level lag varied QBO as predictors. Note the four forecast "busts": 2010-JJA, 2011-JFM,2015-OND and 2016-OND')
    filename = 'MLR_H2O_predict_lon-time_82_cdas-plags_ch4_enso.png'
    plt.savefig(savefig_path / filename, bbox_inches='tight')
    return fg


def plot_figure_7(path=work_chaim, robust=False):
    import xarray as xr
    rds = xr.open_dataset(
        path /
        'MLR_H2O_latpress_seasons_cdas-plags_ch4_enso_1984-2018.nc')
    rds = rds.sortby('season')
    plt_kwargs = {'cmap': predict_cmap, 'figsize': (15, 10),
                  'add_colorbar': False,
                  'extend': None, 'yscale': 'log',
                  'yincrease': False, 'center': 0.0, 'levels': 41}#, 'vmax': vmax}
    fg = rds['params'].plot.contourf(
        col='regressors', row='season', **plt_kwargs, robust=robust)
    cbar_ax = fg.fig.add_axes([0.1, 0.1, .8, .025])
    fg.add_colorbar(
        cax=cbar_ax, orientation="horizontal", label='',
        format='%0.3f')
    fg.fig.subplots_adjust(bottom=0.2, top=0.95, left=0.06)
    [ax.invert_yaxis() for ax in fg.axes.flat]
    [ax.yaxis.set_major_formatter(ScalarFormatter()) for ax in fg.axes.flat]
    for top_ax in fg.axes[0]:
        remove_regressors_and_set_title(top_ax)
    fg.fig.canvas.draw()
    for bottom_ax in fg.axes[-1]:
        change_ticks_lat(fg.fig, bottom_ax, draw=False)
        bottom_ax.set_xlabel('')
    print('Caption: ')
    print('The beta coefficients for the water vapor MLR season analysis for pressure levels vs. latitude with  CH4, ENSO  pressure level lag varied QBO as predictors. This MLR analysis spanned from 1984 to 2018. Note that ENSO is dominant in the MAM season')
    filename = 'MLR_H2O_params_level-lat_seasons_cdas-plags_ch4_enso.png'
    plt.savefig(savefig_path / filename, bbox_inches='tight')
    return fg


def plot_figure_8(path=work_chaim):
    import xarray as xr
    from ML_OOP_stratosphere_gases import plot_like_results
    rds = xr.open_dataset(
        path / 'MLR_H2O_latlon_cdas-plags_ch4_enso_radio_cold_36lags_2004-2018.nc')
    fg = plot_like_results(rds, plot_key='predict_lat-time', level=82,
                           cmap=predict_cmap, extend=None)
    top_ax = fg.axes[0][0]
    mid_ax = fg.axes[1][0]
    bottom_ax = fg.axes[-1][0]
    # remove time from xlabel:
    bottom_ax.set_xlabel('')
    # new ticks:
    ax = change_xticks_years(bottom_ax, start=2008, end=2018)
    fg.fig.tight_layout()
    fg.fig.subplots_adjust(left=0.06, bottom=0.185)
    top_ax.set_title(
        'Area-averaged (from 180W to 180E longitudes) combined H2O anomaly for the 82 hPa pressure level')
    mid_ax.set_title('MLR reconstruction')
    bottom_ax.set_title('Residuals')
    fg.fig.canvas.draw()
    for axes in fg.axes.flatten():
        ax = change_ticks_lat(fg.fig, axes, which_axes='y', draw=False)
        ax.set_ylabel('')
        ax = plot_forecast_busts_lines(ax, color='k')
    print('Caption: ')
    print('The zonal mean water vapor anomalies for the 82 hPa level and their MLR reconstruction and residuals, spanning from 2004 to 2018. This MLR analysis was carried out with CH4 ,ENSO, BDC, T500 and pressure level lag varied QBO as predictors. Note that T500 and BDC predictors were not able to deal with the forecast busts')
    filename = 'MLR_H2O_predict_lat-time_82_cdas-plags_radio_cold36_ch4_enso.png'
    plt.savefig(savefig_path / filename, bbox_inches='tight')
    return fg


def plot_figure_9(path=work_chaim, robust=False):
    from ML_OOP_stratosphere_gases import plot_like_results
    import xarray as xr
    rds = xr.open_dataset(
        path /
        'MLR_H2O_latlon_cdas-plags_ch4_enso_radio_cold_36lags_2004-2018.nc')
    fg = plot_like_results(rds, plot_key='predict_lon-time', level=82,
                           cmap=predict_cmap, robust=robust, extend=None)
    top_ax = fg.axes[0][0]
    mid_ax = fg.axes[1][0]
    bottom_ax = fg.axes[-1][0]
    # remove time from xlabel:
    bottom_ax.set_xlabel('')
    # new ticks:
    ax = change_xticks_years(bottom_ax, start=2008, end=2018)
    fg.fig.tight_layout()
    fg.fig.subplots_adjust(left=0.06, bottom=0.185)
    top_ax.set_title(
        'Area-averaged (weighted by cosine of latitudes 60S to 60N) combined H2O anomaly for the 82 hPa pressure level')
    mid_ax.set_title('MLR reconstruction')
    bottom_ax.set_title('Residuals')
    fg.fig.canvas.draw()
    for axes in fg.axes.flatten():
        ax = change_ticks_lon(fg.fig, axes, which_axes='y', draw=False)
        ax.set_ylabel('')
        ax = plot_forecast_busts_lines(ax, color='k')
    print('Caption: ')
    print('The meridional mean water vapor anomalies for the 82 hPa level and their MLR reconstruction and residuals, spanning from 2004 to 2018. This MLR analysis was carried out with CH4 ,ENSO, BDC, T500 and pressure level lag varied QBO as predictors. Note that T500 and BDC predictors were not able to deal with the forecast busts')
    filename = 'MLR_H2O_predict_lon-time_82_cdas-plags_radio_cold36_ch4_enso.png'
    plt.savefig(savefig_path / filename, bbox_inches='tight')
    return fg


def plot_figure_10(path=work_chaim):
    import xarray as xr
    from ML_OOP_stratosphere_gases import plot_like_results
    rds = xr.open_dataset(
        path / 'MLR_H2O_latlon_cdas-plags_ch4_enso-plags_t500-plags_bdc-plags_2004-2018.nc')
    fg = plot_like_results(rds, plot_key='predict_lat-time', level=82,
                           cmap=predict_cmap, extend=None)
    top_ax = fg.axes[0][0]
    mid_ax = fg.axes[1][0]
    bottom_ax = fg.axes[-1][0]
    # remove time from xlabel:
    bottom_ax.set_xlabel('')
    # new ticks:
    ax = change_xticks_years(bottom_ax, start=2005, end=2018)
    fg.fig.tight_layout()
    fg.fig.subplots_adjust(left=0.06, bottom=0.185)
    top_ax.set_title(
        'Area-averaged (from 180W to 180E longitudes) combined H2O anomaly for the 82 hPa pressure level')
    mid_ax.set_title('MLR reconstruction')
    bottom_ax.set_title('Residuals')
    fg.fig.canvas.draw()
    for axes in fg.axes.flatten():
        ax = change_ticks_lat(fg.fig, axes, which_axes='y', draw=False)
        ax.set_ylabel('')
        ax = plot_forecast_busts_lines(ax, color='k')
    print('Caption: ')
    print('The zonal mean water vapor anomalies for the 82 hPa level and their MLR reconstruction and residuals, spanning from 2004 to 2018. This MLR analysis was carried out with CH4 ,ENSO, BDC, T500 and pressure level lag varied QBO as predictors. Note that T500 and BDC predictors were not able to deal with the forecast busts')
    filename = 'MLR_H2O_predict_lat-time_82_cdas-plags_t500_plags_bdc_plags_ch4_enso.png'
    plt.savefig(savefig_path / filename, bbox_inches='tight')
    return fg


def plot_figure_11(path=work_chaim, robust=False):
    from ML_OOP_stratosphere_gases import plot_like_results
    import xarray as xr
    rds = xr.open_dataset(
        path /
        'MLR_H2O_latlon_cdas-plags_ch4_enso-plags_t500-plags_bdc-plags_2004-2018.nc')
    fg = plot_like_results(rds, plot_key='predict_lon-time', level=82,
                           cmap=predict_cmap, robust=robust, extend=None)
    top_ax = fg.axes[0][0]
    mid_ax = fg.axes[1][0]
    bottom_ax = fg.axes[-1][0]
    # remove time from xlabel:
    bottom_ax.set_xlabel('')
    # new ticks:
    ax = change_xticks_years(bottom_ax, start=2005, end=2018)
    fg.fig.tight_layout()
    fg.fig.subplots_adjust(left=0.06, bottom=0.185)
    top_ax.set_title(
        'Area-averaged (weighted by cosine of latitudes 60S to 60N) combined H2O anomaly for the 82 hPa pressure level')
    mid_ax.set_title('MLR reconstruction')
    bottom_ax.set_title('Residuals')
    fg.fig.canvas.draw()
    for axes in fg.axes.flatten():
        ax = change_ticks_lon(fg.fig, axes, which_axes='y', draw=False)
        ax.set_ylabel('')
        ax = plot_forecast_busts_lines(ax, color='k')
    print('Caption: ')
    print('The meridional mean water vapor anomalies for the 82 hPa level and their MLR reconstruction and residuals, spanning from 2004 to 2018. This MLR analysis was carried out with CH4 ,ENSO, BDC, T500 and pressure level lag varied QBO as predictors. Note that T500 and BDC predictors were not able to deal with the forecast busts')
    filename = 'MLR_H2O_predict_lon-time_82_cdas-plags_t500_plags_bdc_plags_ch4_enso.png'
    plt.savefig(savefig_path / filename, bbox_inches='tight')
    return fg


def plot_figure_12(path=work_chaim, rds=None, save=True):
    """r2 map (lat-lon) for cdas-plags, enso, ch4"""
    import xarray as xr
    import cartopy.crs as ccrs
    import numpy as np
    from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
    if rds is None:
        rds = xr.open_dataset(
                path /
                'MLR_H2O_latlon_cdas-plags_ch4_enso_2004-2018.nc')
    rds = rds['r2_adj'].sel(level=82, method='nearest')
    fig = plt.figure(figsize=(11, 5))
    ax = fig.add_subplot(1, 1, 1,
                         projection=ccrs.PlateCarree(central_longitude=0))
    ax.coastlines()
    fg = rds.plot.contourf(ax=ax, add_colorbar=False, cmap=error_cmap,
                           vmin=0.0, extend=None, levels=21)
    ax.set_title('')
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
    cbar_kws = {'label': '', 'format': '%0.2f'}
    cbar_ax = fg.ax.figure.add_axes([0.1, 0.1, .8, .035])
    plt.colorbar(fg, cax=cbar_ax, orientation="horizontal", **cbar_kws)
    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        linewidth=1,
        color='black',
        alpha=0.5,
        linestyle='--',
        draw_labels=True)
    gl.xlines = True
    gl.xlocator = mticker.FixedLocator([-180, -120, -60, 0, 60, 120, 180])
    gl.ylocator = mticker.FixedLocator([-45, -30, -15, 0, 15, 30 ,45])
    gl.xlabel_style = {'size': 10}
    gl.ylabel_style = {'size': 10}
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    fig.tight_layout()
    fig.subplots_adjust(top=0.98,left=0.06, right=0.94)
    print('Caption: ')
    print('The adjusted R^2 for the water vapor anomalies MLR analysis in the 82 hPa level with CH4 ,ENSO, and pressure level lag varied QBO as predictors. This MLR spans from 2004 to 2018')
    filename = 'MLR_H2O_r2_map_82_cdas-plags_ch4_enso.png'
    if save:
        plt.savefig(savefig_path / filename, bbox_inches='tight')
    return fg


def plot_figure_13(path=work_chaim, rds=None, save=True):
    """params map (lat-lon) for cdas-plags, enso, ch4"""
    import xarray as xr
    import cartopy.crs as ccrs
    from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
    if rds is None:
        rds = xr.open_dataset(
            path /
            'MLR_H2O_latlon_cdas-plags_ch4_enso_2004-2018.nc')
    rds = rds['params'].sel(level=82, method='nearest')
    proj = ccrs.PlateCarree(central_longitude=0)
#    fig, axes = plt.subplots(1, 3, figsize=(17, 3.0),
#                             subplot_kw=dict(projection=proj))
    gl_list = []
    fg = rds.plot.contourf(col='regressors', add_colorbar=False,
                           cmap=predict_cmap, center=0.0, extend=None,
                           levels=41, subplot_kws=dict(projection=proj),
                           transform=ccrs.PlateCarree(), figsize=(17, 3))
    cbar_kws = {'label': '', 'format': '%0.2f'}
    cbar_ax = fg.fig.add_axes([0.1, 0.1, .8, .035])  # last num controls width
    fg.add_colorbar(cax=cbar_ax, orientation="horizontal", **cbar_kws)
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
    gl_list[0].ylabels_right = False
    gl_list[2].ylabels_left = False
    fg.fig.tight_layout()
    fg.fig.subplots_adjust(right=0.96, left=0.04, wspace=0.15)
    print('Caption: ')
    print('The beta coeffciants for the water vapor anomalies MLR analysis in the 82 hPa level at 2004 to 2018')
    filename = 'MLR_H2O_params_map_82_cdas-plags_ch4_enso.png'
    if save:
        plt.savefig(savefig_path / filename, bbox_inches='tight')
    return fg


def plot_figure_seasons(path=work_chaim, rds=None, field='r2_adj', level=82,
                        save=True, add_to_suptitle=None):
    import xarray as xr
    import cartopy.crs as ccrs
    if field == 'r2_adj':
        cmap = error_cmap
        center = None
        vmin = 0.0
        col = 'season'
        row = None
        figsize = (17, 3)
    elif field == 'params':
        cmap = predict_cmap
        vmin = None
        center = 0.0
        col = 'regressors'
        row = 'season'
        figsize = (17, 17)
    proj = ccrs.PlateCarree(central_longitude=0)
    if rds is None:
        rds = xr.open_dataset(
            path /
            '')
    rds = rds[field].sel(level=level, method='nearest')
    fg = rds.plot.contourf(col=col, row=row, add_colorbar=False,
                           cmap=cmap, center=center, vmin=vmin, extend=None,
                           levels=41, subplot_kws=dict(projection=proj),
                           transform=ccrs.PlateCarree(), figsize=figsize)
    fg = add_horizontal_colorbar(fg, width=0.025, cbar_kwargs_dict=None)
    [ax.coastlines() for ax in fg.axes.flatten()]
    [ax.gridlines(
        crs=ccrs.PlateCarree(),
        linewidth=1,
        color='black',
        alpha=0.5,
        linestyle='--',
        draw_labels=False) for ax in fg.axes.flatten()]
    filename = 'MLR_H2O_{}_map_{}_cdas-plags_ch4_enso_seasons.png'.format(
        field, level)
    sup = '{} for the {} hPa level'.format(fields_dict[field], level)
    if add_to_suptitle is not None:
        sup += add_to_suptitle
    fg.fig.suptitle(sup)
    if field == 'params':
        fg.fig.subplots_adjust(bottom=0.11, top=0.95)
    if save:
        plt.savefig(savefig_path / filename, bbox_inches='tight')
    return fg
