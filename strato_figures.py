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
from palettable.scientific import sequential as seqsci
from palettable.colorbrewer import sequential as seqbr
from palettable.scientific import diverging as divsci
from palettable.colorbrewer import diverging as divbr
from matplotlib.colors import ListedColormap
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

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


def plot_forecast_busts_lines(ax, color='r', style='--'):
    ax.axvline('2010-05', c=color, ls=style)
    ax.axvline('2010-09', c=color, ls=style)
    ax.axvline('2010-12', c=color, ls=style)
    ax.axvline('2011-04', c=color, ls=style)
    ax.axvline('2015-09', c=color, ls=style)
    ax.axvline('2016-01', c=color, ls=style)
    ax.axvline('2016-09', c=color, ls=style)
    ax.axvline('2017-01', c=color, ls=style)
    return ax

def remove_regressors_and_set_title(ax):
    short_titles = {'qbo_cdas': 'QBO',
                    'anom_nino3p4': 'ENSO',
                    'ch4': 'CH4',
                    'era5_bdc': 'BDC',
                    'era5_t500': 'T at 500hPa'}
    title = ax.get_title()
    title = title.split('=')[-1].strip(' ')
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
            xlabel = '{}S'.format(str(label).split('-')[-1])
        elif label > 0:
            xlabel = '{}N'.format(label)
        elif label == 0:
            xlabel = '0'
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
            xlabel = '{}W'.format(str(label).split('-')[-1])
        elif label > 0:
            xlabel = '{}E'.format(label)
        elif label == 0:
            xlabel = '0'
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
    return fg


def plot_figure_2(path=work_chaim, robust=False):
    from ML_OOP_stratosphere_gases import plot_like_results
    import xarray as xr
    rds = xr.open_dataset(
        work_chaim /
        'MLR_H2O_latpress_cdas-plags_ch4_enso_1984-2018.nc')
    fg = plot_like_results(rds, plot_key='predict_level-time', lat=None,
                           cmap=predict_cmap, robust=robust)
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
    return fg


def plot_figure_3(path=work_chaim, robust=False):
    from ML_OOP_stratosphere_gases import plot_like_results
    import xarray as xr
    rds = xr.open_dataset(
        work_chaim /
        'MLR_H2O_latpress_cdas-plags_ch4_enso_1984-2018.nc')
    fg = plot_like_results(rds, plot_key='r2_level-lat', cmap=error_cmap,
                           robust=robust)
    fg.ax.set_title('')
    change_ticks_lat(fg.ax.figure, fg.ax)
    fg.ax.figure.tight_layout()
    fg.ax.figure.subplots_adjust(left=0.15, right=0.95)
    fg.ax.set_xlabel(r'Latitude [$\degree$]')
    print('Caption: ')
    print('The adjusted R^2 for the water vapor MLR analysis(1984-2018) with CH4, ENSO and pressure level lag varied QBO as predictors')
    return fg


def plot_figure_4(path=work_chaim, robust=False):
    from ML_OOP_stratosphere_gases import plot_like_results
    import xarray as xr
    rds = xr.open_dataset(
        work_chaim /
        'MLR_H2O_latpress_cdas-plags_ch4_enso_1984-2018.nc')
    fg = plot_like_results(rds, plot_key='params_level-lat', cmap=predict_cmap,
                           figsize=(10, 5), robust=robust)
    fg.fig.suptitle('')
    fg.fig.canvas.draw()
    for ax in fg.axes.flatten():
        change_ticks_lat(fg.fig, ax, draw=False)
        ax.set_xlabel(r'Latitude [$\degree$]')
        ax = remove_regressors_and_set_title(ax)
    fg.fig.tight_layout()
    fg.fig.subplots_adjust(left=0.06, bottom=0.25)
    print('Caption: ')
    print('The beta coefficiants for the water vapor MLR analysis(1984-2018) with CH4, ENSO and pressure level lag varied QBO as predictors')
    return fg


def plot_figure_5(path=work_chaim, robust=False):
    from ML_OOP_stratosphere_gases import plot_like_results
    import xarray as xr
    rds = xr.open_dataset(
        work_chaim /
        'MLR_H2O_latlon_cdas-plags_ch4_enso_2004-2018.nc')
    fg = plot_like_results(rds, plot_key='predict_lat-time', level=82,
                           cmap=predict_cmap, robust=robust)
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
        ax.set_ylabel(r'Latitude [$\degree$]')
        ax = plot_forecast_busts_lines(ax, color='k')
    print('Caption: ')
    print('The zonal mean water vapor anomalies for the 82 hPa level and their MLR reconstruction and residuals, spanning from 2004 to 2018. This MLR analysis was carried out with CH4 ,ENSO and pressure level lag varied QBO as predictors. Note the four forecast "busts": 2010-JJA, 2011-JFM,2015-OND and 2016-OND')
    return fg


def plot_figure_6(path=work_chaim, robust=False):
    from ML_OOP_stratosphere_gases import plot_like_results
    import xarray as xr
    rds = xr.open_dataset(
        work_chaim /
        'MLR_H2O_latlon_cdas-plags_ch4_enso_2004-2018.nc')
    fg = plot_like_results(rds, plot_key='predict_lon-time', level=82,
                           cmap=predict_cmap, robust=robust)
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
        ax.set_ylabel(r'Longitude [$\degree$]')
        ax = plot_forecast_busts_lines(ax, color='k')
    print('Caption: ')
    print('The meridional mean water vapor anomalies for the 82 hPa level and their MLR reconstruction and residuals, spanning from 2004 to 2018. This MLR analysis was carried out with CH4 ,ENSO and pressure level lag varied QBO as predictors. Note the four forecast "busts": 2010-JJA, 2011-JFM,2015-OND and 2016-OND')
    return fg


def plot_figure_7(path=work_chaim, robust=False):
    import xarray as xr
    rds = xr.open_dataset(
        work_chaim /
        'MLR_H2O_latpress_seasons_cdas-plags_ch4_enso_1984-2018.nc')
    rds = rds.sortby('season')
    vmax = rds.params.max()
    plt_kwargs = {'cmap': predict_cmap, 'figsize': (15, 10),
                  'add_colorbar': False,
                  'extend': 'both', 'yscale': 'log',
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
        bottom_ax.set_xlabel(r'Latitude [$\degree$]')
    print('Caption: ')
    print('The beta coefficients for the water vapor MLR season analysis for pressure levels vs. latitude with  CH4, ENSO  pressure level lag varied QBO as predictors. This MLR analysis spanned from 1984 to 2018. Note that ENSO is dominant in the MAM season')
    return fg
