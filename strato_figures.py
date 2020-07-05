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
import matplotlib.ticker as ticker
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


def add_horizontal_colorbar(fg_obj, rect=[0.1, 0.1, 0.8, 0.025], cbar_kwargs_dict=None):
    # rect = [left, bottom, width, height]
    # add option for just figure object, now, accepts facetgrid object only
    cbar_kws = {'label': '', 'format': '%0.2f'}
    if cbar_kwargs_dict is not None:
        cbar_kws.update(cbar_kwargs_dict)
    cbar_ax = fg_obj.fig.add_axes(rect)
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


#def add_season_equals(ax):
#    if ax.texts:
#        # This contains the right ylabel text
#        txt = ax.texts[0]
#        label = txt.get_text()
#        label = 'season = {}'.format(label)
#        ax.text(txt.get_unitless_position()[0], txt.get_unitless_position()[1],
#                label,
#                transform=ax.transAxes,
#                va='center',
#                # fontsize='xx-large',
#                rotation=-90)
#        # Remove the original text
#        ax.texts[0].remove()
#    return ax


def remove_time_and_set_date(ax):
    if ax.texts:
    # This contains the right ylabel text
        txt = ax.texts[0]
        label = txt.get_text()
        label = '-'.join(label.split('=')[-1].strip(' ').split('-')[0:2])
        ax.text(txt.get_unitless_position()[0], txt.get_unitless_position()[1],
                label,
                transform=ax.transAxes,
                va='center',
                # fontsize='xx-large',
                rotation=-90)
        # Remove the original text
        ax.texts[0].remove()
    return ax


def remove_anomaly_and_set_title(ax, species='H2O'):
    if species == 'H2O':
        original = 'Combined water vapor'
    elif species == 't':
        original = 'Air temperature'
    elif species == 'u':
        original = 'Zonal wind'
    short_titles = {'original': '{} anomaly'.format(original),
                    'predict': 'MLR reconstruction',
                    'resid': 'Residuals'}
    title = ax.get_title()
    title = title.split('=')[-1].strip(' ')
    title = short_titles.get(title)
    ax.set_title(title)
    return ax


def remove_regressors_and_set_title(ax, set_title_only=None):
    short_titles = {'qbo_cdas': 'QBO',
                    'anom_nino3p4': 'ENSO',
                    'ch4': 'CH4',
                    'era5_bdc': 'BDC',
                    'era5_t500': 'T at 500hPa',
                    'anom_nino3p4^2': r'ENSO$^2$',
                    'anom_nino3p4*q...': r'ENSO $\times$ QBO',
                    '1': r'(QBO + ENSO + CH4) = $\eta_1$',
                    '2': r'$\eta_1$ + T500 + BDC',
                    '3': r'$\eta_1$ + $\sum^6_{i=0}$CPT(t-$i$)',
                    '4': r'$\eta_1$ + QBO $\times$ ENSO + ENSO$^2$'}
    title = ax.get_title()
    title = title.split('=')[-1].strip(' ')
    if set_title_only is not None:
        title = short_titles.get(set_title_only)
    else:
        title = short_titles.get(title)
    ax.set_title(title)
    return ax


@ticker.FuncFormatter
def lon_formatter(x, pos):
    if x < 0:
        return r'{0:.0f}$\degree$W'.format(abs(x))
    elif x > 0:
        return r'{0:.0f}$\degree$E'.format(abs(x))
    elif x == 0:
        return r'0$\degree$'

@ticker.FuncFormatter
def lat_formatter(x, pos):
    if x < 0:
        return r'{0:.0f}$\degree$S'.format(abs(x))
    elif x > 0:
        return r'{0:.0f}$\degree$N'.format(abs(x))
    elif x == 0:
        return r'0$\degree$'


@ticker.FuncFormatter
def single_digit_formatter(x, pos):
    return '{0:.0f}'.format(x)


def change_xticks_years(ax, start=1984, end=2018):
    import pandas as pd
    import numpy as np
    years_fmt = mdates.DateFormatter('%Y')
    years = np.arange(start, end + 1, 1)
    years = [pd.to_datetime(str(x)).strftime('%Y') for x in years]
    ax.set_xticks(years)
    # ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(years_fmt)
    return ax


def plot_reg_correlation_heatmap(
        regs=['qbo_cdas', 'anom_nino3p4', 'era5_bdc70', 'era5_bdc82',
              'era5_t500', 'radio_cold', 'radio_cold_no_qbo'],
        lms=None, plevel=82, time_period=['1984', '2019']):
    import seaborn as sns
    from ML_OOP_stratosphere_gases import PredictorSet
    Pset = PredictorSet(regressors=regs,time_period=time_period,
                        loadpath=Path().cwd()/'regressors')
    ds = Pset.pre_process(stack=False)
    if lms is not None:
        lms = lms.sel(level=plevel, method='nearest')
        lms = lms.where(lms!=0).dropna('reg_shifted').astype(int)
        regs_to_shift = list(
            set(lms.reg_shifted.values).intersection(set([x for x in ds])))
        for reg in regs_to_shift:
            lag = lms.sel(reg_shifted=reg).values.item()
            ds[reg] = ds[reg].shift(time=lag)
            ds = ds.rename({reg: '{}({})'.format(reg, lag)})
            ds.attrs[reg] = lag
        ds.attrs['level_shifted'] = plevel
    df = ds.to_dataframe()
    g = sns.heatmap(df.corr(),annot=True,cmap='bwr',center=0)
    g.set_xticklabels(g.get_xticklabels(), rotation = 45, fontsize = 12)
    if lms is not None:
        g.set_title('{}-{} with lags in parenthatis that maximize R$^2$ H2O in {} hPa'.format(time_period[0], time_period[1], plevel))
    else:
        g.set_title('{}-{}'.format(time_period[0], time_period[1]))
    g.figure.tight_layout()
    return g


def plot_figure_1(path=work_chaim, regressors=['qbo_cdas']):
    from ML_OOP_stratosphere_gases import run_ML
    # sns.set_style('ticks', rc=rc)
    cbar_kws = {'label': '', 'format': '%0.2f', 'aspect': 50}
    if len(regressors) == 1:
        rds = run_ML(time_period=['1984', '2019'], regressors=regressors,
                     special_run={'optimize_reg_shift': [0, 12]},
                     area_mean=True, lat_slice=[-20, 20])
        fg = rds.r2_adj.T.plot.pcolormesh(yscale='log', yincrease=False,
                                          levels=21, col='reg_shifted',
                                          cmap=error_cmap, extend=None,
                                          figsize=(7, 7),
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
        'MLR_H2O_latpress_cdas-plags_ch4_enso_1984-2019.nc')
    fg = plot_like_results(rds, plot_key='predict_level-time', lat=None,
                           cmap=predict_cmap, robust=robust, extend=None,
                           no_colorbar=True)
    top_ax = fg.axes[0][0]
    mid_ax = fg.axes[1][0]
    bottom_ax = fg.axes[-1][0]
    # remove time from xlabel:
    bottom_ax.set_xlabel('')
    # new ticks:
    bottom_ax = change_xticks_years(bottom_ax, start=1985, end=2019)
    top_ax.set_title(
        r'Area-averaged (weighted by cosine of latitudes 60$\degree$S to 60$\degree$N) combined water vapor anomaly')
    mid_ax.set_title('MLR reconstruction')
    bottom_ax.set_title('Residuals')
    [ax.yaxis.set_major_formatter(single_digit_formatter)
     for ax in [top_ax, mid_ax, bottom_ax]]
    fg = add_horizontal_colorbar(fg, [0.125, 0.057, 0.8, 0.02],
                                 cbar_kwargs_dict={'label': 'ppmv'})
    fg.fig.tight_layout()
    fg.fig.subplots_adjust(left=0.06, bottom=0.14)
    print('Caption: ')
    print('Stratospheric water vapor anomalies and their MLR reconstruction and residuals, spanning from 1984 to 2018 and using CH4, ENSO and pressure level lag varied QBO as predictors')
    filename = 'MLR_H2O_predict_level-time_cdas-plags_ch4_enso.png'
    plt.savefig(savefig_path / filename, bbox_inches='tight')
    return fg


def plot_figure_3(path=work_chaim):
    from ML_OOP_stratosphere_gases import plot_like_results
    import xarray as xr
    rds = xr.open_dataset(
        path /
        'MLR_H2O_latpress_cdas-plags_ch4_enso_1984-2019.nc')
    fg = plot_like_results(rds, plot_key='r2_level-lat', cmap=error_cmap,
                           extend=None, add_colorbar=True)
    fg.ax.set_title('')
    fg.ax.xaxis.set_major_formatter(lat_formatter)
    fg.ax.yaxis.set_major_formatter(single_digit_formatter)
    fg.ax.figure.tight_layout()
    fg.ax.figure.subplots_adjust(left=0.15, right=1.0, bottom=0.05, )
    fg.ax.set_xlabel('')
    print('Caption: ')
    print('The adjusted R^2 for the water vapor MLR analysis(1984-2018) with CH4, ENSO and pressure level lag varied QBO as predictors')
    filename = 'MLR_H2O_r2_level-lat_cdas-plags_ch4_enso.png'
    plt.savefig(savefig_path / filename, bbox_inches='tight')
    return fg


def plot_figure_4(path=work_chaim):
    from ML_OOP_stratosphere_gases import plot_like_results
    import xarray as xr
    rds = xr.open_dataset(
        path /
        'MLR_H2O_latpress_cdas-plags_ch4_enso_1984-2019.nc')
    fg = plot_like_results(rds, plot_key='params_level-lat', cmap=predict_cmap,
                           figsize=(10, 5), extend=None, no_colorbar=True)
    fg.fig.suptitle('')
    fg.fig.canvas.draw()
    for ax in fg.axes.flatten():
        ax.yaxis.set_major_formatter(single_digit_formatter)
        ax.xaxis.set_major_formatter(lat_formatter)
        ax.set_xlabel('')
        ax = remove_regressors_and_set_title(ax)
        
    fg = add_horizontal_colorbar(fg, [0.125, 0.1, 0.8, 0.02],
                                 cbar_kwargs_dict={'label': ''})
    fg.fig.tight_layout()
    fg.fig.subplots_adjust(left=0.1, bottom=0.21)
    print('Caption: ')
    print('The beta coefficiants for the water vapor MLR analysis(1984-2018) with CH4, ENSO and pressure level lag varied QBO as predictors')
    filename = 'MLR_H2O_params_level-lat_cdas-plags_ch4_enso.png'
    plt.savefig(savefig_path / filename, bbox_inches='tight')
    return fg


def plot_latlon_predict(ncfile, path=work_chaim, geo='lat', level=82.54,
                        bust_lines=True, st_year=None, lat_slice=[-20, 20],
                        save=True):
    from ML_OOP_stratosphere_gases import plot_like_results
    import xarray as xr
    import math
    rds = xr.load_dataset(path / ncfile)
    if lat_slice is not None:
        rds = rds.sel(lat=slice(*lat_slice))
    species = ncfile.split('.')[0].split('_')[1]
    regs = '_'.join(ncfile.split('.')[0].split('_')[3: -1])
    if species == 'H2O':
        geo_title = {
                'lat': r'Area-averaged (from 180$\degree$W to 180$\degree$E longitudes) combined H2O anomaly for the {} hPa pressure level'.format(level),
                'lon': r'Area-averaged (weighted by cosine of latitudes 60$\degree$S to 60$\degree$N) combined H2O anomaly for the {} hPa pressure level'.format(level)}
        if st_year is None:
            st_year = 2005
        unit = 'ppmv'
    elif species == 't':
        geo_title = {
                'lat': r'Area-averaged (from 180$\degree$W to 180$\degree$E longitudes) air temperature anomaly for the {} hPa pressure level'.format(level),
                'lon': r'Area-averaged (weighted by cosine of latitudes 60$\degree$S to 60$\degree$N) air temperature anomaly for the {} hPa pressure level'.format(level)}
        st_year = 1984
        unit = 'K'
    elif species == 'u':
        geo_title = {
                'lat': r'Area-averaged (from 180$\degree$W to 180$\degree$E longitudes) zonal wind anomaly for the {} hPa pressure level'.format(level),
                'lon': r'Area-averaged (weighted by cosine of latitudes 60$\degree$S to 60$\degree$N) zonal wind anomaly for the {} hPa pressure level'.format(level)}
        st_year = 1984
        unit = r'm$\cdot$sec$^{-1}$'
    fg = plot_like_results(rds, plot_key='predict_{}-time'.format(geo),
                           level=level, cmap=predict_cmap, extend=None,
                           no_colorbar=True)
    top_ax = fg.axes[0][0]
    mid_ax = fg.axes[1][0]
    bottom_ax = fg.axes[-1][0]
    # remove time from xlabel:
    bottom_ax.set_xlabel('')
    # new ticks:
    # bottom_ax = change_xticks_years(bottom_ax, start=st_year, end=2019)
    top_ax.set_title(geo_title.get(geo))
    mid_ax.set_title('MLR reconstruction')
    bottom_ax.set_title('Residuals')
    # fg.fig.canvas.draw()
    for ax in [top_ax, mid_ax, bottom_ax]:
        if geo == 'lon':
            ax.yaxis.set_major_formatter(lon_formatter)
        elif geo == 'lat':
            ax.yaxis.set_major_formatter(lat_formatter)
        ax.set_ylabel('')
        if bust_lines:
            ax = plot_forecast_busts_lines(ax, color='k')
    fg = add_horizontal_colorbar(fg, [0.1, 0.065, 0.8, 0.015],
                             cbar_kwargs_dict={'label': unit})
    fg.fig.tight_layout()
    fg.fig.subplots_adjust(left=0.05, bottom=0.14)
    if save:
        filename = 'MLR_{}_predict_{}-time_{}_{}_{}-2019.png'.format(species, geo, math.floor(level), regs, st_year)
        fg.fig.savefig(savefig_path / filename , bbox_inches='tight')
    return fg


def plot_figure_5(path=work_chaim):
    ncfile = 'MLR_H2O_latlon_cdas-plags_ch4_enso_2004-2019.nc'
    fg = plot_latlon_predict(ncfile, path=path, geo='lat', level=82.54,
                             bust_lines=True, save=True)
    print('Caption: ')
    print('The zonal mean water vapor anomalies for the 82.54 hPa level and their MLR reconstruction and residuals, spanning from 2004 to 2018. This MLR analysis was carried out with CH4 ,ENSO and pressure level lag varied QBO as predictors. Note the three forecast "busts": 2010-D to 2011-JFM, 2015-OND and 2016-OND')
    return fg


def plot_figure_6(path=work_chaim):
    ncfile = 'MLR_H2O_latlon_cdas-plags_ch4_enso_2004-2019.nc'
    fg = plot_latlon_predict(ncfile, path=path, geo='lon', level=82.54,
                             bust_lines=True, save=True)
    print('Caption: ')
    print('The meridional mean water vapor anomalies for the 82.54 hPa level and their MLR reconstruction and residuals, spanning from 2004 to 2018. This MLR analysis was carried out with CH4 ,ENSO and pressure level lag varied QBO as predictors. Note the three forecast "busts": 2010-D to 2011-JFM, 2015-OND and 2016-OND')
    return fg


def plot_figure_seasons(ncfile, path=work_chaim, field='params', reduce=None,
                        plot_kwargs=None):
    import xarray as xr
    rds = xr.open_dataset(path / ncfile)
    species = ncfile.split('.')[0].split('_')[1]
    if species == 'H2O':
        unit = 'ppmv'
    elif species == 't':
        unit = 'K'
    elif species == 'u':
        unit = r'm$\cdot$sec$^{-1}$'
    if field == 'params':
        unit = ''
    regs = '_'.join(ncfile.split('.')[0].split('_')[3: -1])
    syear = ncfile.split('.')[0].split('_')[-1].split('-')[0]
    eyear = ncfile.split('.')[0].split('_')[-1].split('-')[-1]
    rds = rds.sortby('season')
    if reduce is not None:
        rds = rds.mean(reduce, keep_attrs=True)
    plt_kwargs = {'cmap': predict_cmap, 'figsize': (15, 10),
                  'add_colorbar': False,
                  'extend': None, 'yscale': 'log',
                  'yincrease': False, 'center': 0.0, 'levels': 41}
    if plot_kwargs is not None:
        plt_kwargs.update(plot_kwargs)
    data = rds[field]
    fg = data.plot.contourf(
        col='regressors', row='season', **plt_kwargs)
    fg = add_horizontal_colorbar(fg, [0.1, 0.065, 0.8, 0.015], cbar_kwargs_dict={'label': unit})
    fg.fig.subplots_adjust(bottom=0.13, top=0.95, left=0.06)
    [ax.invert_yaxis() for ax in fg.axes.flat]
    [ax.yaxis.set_major_formatter(ScalarFormatter()) for ax in fg.axes.flat]
    [ax.yaxis.set_major_formatter(single_digit_formatter)
     for ax in fg.axes.flat]
    for top_ax in fg.axes[0]:
        remove_regressors_and_set_title(top_ax)
    fg.fig.canvas.draw()
    for bottom_ax in fg.axes[-1]:
        bottom_ax.xaxis.set_major_formatter(lat_formatter)
        bottom_ax.set_xlabel('')
    filename = 'MLR_{}_params_level-lat_{}_{}-{}.png'.format(species, regs, syear, eyear)
    plt.savefig(savefig_path / filename, bbox_inches='tight')
    return fg


def plot_figure_7(path=work_chaim):
    ncfile = 'MLR_H2O_latpress_seasons_cdas-plags_ch4_enso_1984-2019.nc'
    fg = plot_figure_seasons(ncfile, path, field='params')
    print('Caption: ')
    print('The beta coefficients for the water vapor MLR season analysis for pressure levels vs. latitude with  CH4, ENSO  pressure level lag varied QBO as predictors. This MLR analysis spanned from 1984 to 2018. Note that ENSO is dominant in the MAM season')  
    return fg


def plot_figure_8(path=work_chaim):
    ncfile = 'MLR_H2O_latlon_cdas-plags_ch4_enso_radio_cold_lags6_2004-2019.nc'
    fg = plot_latlon_predict(ncfile, path=path, geo='lat', level=82.54,
                             bust_lines=True, save=True)
    print('Caption: ')
    print('The zonal mean water vapor anomalies for the 82.54 hPa level and their MLR reconstruction and residuals, spanning from 2004 to 2018. This MLR analysis was carried out with CH4 ,ENSO, radio cpt with 6 months lags and pressure level lag varied QBO as predictors. Note that the radio cpt predictor and its 6 lags were able to deal with the forecast busts succsesfully')
    return fg


def plot_figure_9(path=work_chaim):
    ncfile = 'MLR_H2O_latlon_cdas-plags_ch4_enso_radio_cold_lags6_2004-2019.nc'
    fg = plot_latlon_predict(ncfile, path=path, geo='lon', level=82.54,
                             bust_lines=True, save=True)
    print('Caption: ')
    print('The meridional mean water vapor anomalies for the 82.54 hPa level and their MLR reconstruction and residuals, spanning from 2004 to 2018. This MLR analysis was carried out with CH4 ,ENSO, radio cpt with 6 months lags and pressure level lag varied QBO as predictors. Note that the radio cpt predictor and its 6 lags were able to deal with the forecast busts succsesfully')
    return fg


def plot_figure_10(path=work_chaim):
    ncfile = 'MLR_H2O_latlon_cdas-plags_ch4_enso_bdc_t500_2004-2019.nc'
    fg = plot_latlon_predict(ncfile, path=path, geo='lat', level=82.54,
                             bust_lines=True, save=True)
    print('Caption: ')
    print('The zonal mean water vapor anomalies for the 82.54 hPa level and their MLR reconstruction and residuals, spanning from 2004 to 2018. This MLR analysis was carried out with CH4 ,ENSO, BDC, T500 and pressure level lag varied QBO as predictors. Note that T500 and BDC predictors were not able to deal with the forecast busts')
    return fg


def plot_figure_11(path=work_chaim):
    ncfile = 'MLR_H2O_latlon_cdas-plags_ch4_enso_bdc_t500_2004-2019.nc'
    fg = plot_latlon_predict(ncfile, path=path, geo='lon', level=82.54,
                             bust_lines=True, save=True)
    print('Caption: ')
    print('The merdional mean water vapor anomalies for the 82.54 hPa level and their MLR reconstruction and residuals, spanning from 2004 to 2018. This MLR analysis was carried out with CH4 ,ENSO, BDC, T500 and pressure level lag varied QBO as predictors. Note that T500 and BDC predictors were not able to deal with the forecast busts')
    return fg


def plot_r2_map_predictor_sets(path=work_chaim, save=True):
    """r2 map (lat-lon) for cdas-plags, enso, ch4"""
    import xarray as xr
    import cartopy.crs as ccrs
    import numpy as np
    from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
    rds1 = xr.open_dataset(
                path /
                'MLR_H2O_latlon_cdas-plags_ch4_enso_2004-2019.nc')
    rds2 = xr.open_dataset(
            path /
            'MLR_H2O_latlon_cdas-plags_ch4_enso_bdc_t500_2004-2019.nc')
    rds3 = xr.open_dataset(
            path /
            'MLR_H2O_latlon_cdas-plags_ch4_enso_radio_cold_lags6_2004-2019.nc')
    rds4 = xr.open_dataset(
        path /
        'MLR_H2O_latlon_cdas-plags_ch4_enso_poly_2_no_qbo^2_no_ch4_extra_2004-2019.nc')
    rds = xr.concat([x['r2_adj'].sel(level=82, method='nearest') for x in [rds1, rds2, rds3, rds4]], 'eta')
    rds['eta'] = range(1, 5)
    rds = rds.sortby('eta')
#    fig = plt.figure(figsize=(11, 5))
#    ax = fig.add_subplot(1, 1, 1,
#                         projection=ccrs.PlateCarree(central_longitude=0))
#    ax.coastlines()
    proj = ccrs.PlateCarree(central_longitude=0)
    fg = rds.plot.contourf(col='eta', add_colorbar=False, cmap=error_cmap,
                           vmin=0.0, extend=None, levels=21, col_wrap=2,
                           subplot_kws=dict(projection=proj),
                           transform=ccrs.PlateCarree(), figsize=(11, 5))
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
    filename = 'MLR_H2O_r2_map_82_eta.png'
    if save:
        plt.savefig(savefig_path / filename, bbox_inches='tight')
    return fg


def plot_feature_map(ncfile, path=work_chaim, rds=None, feature='params',
                     level=82, col_wrap=3, figsize=(17, 3), extent=[-170, 170, -57.5, 57.5]):
    import xarray as xr
    import cartopy.crs as ccrs
    from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
    if rds is None:
        species = ncfile.split('.')[0].split('_')[1]
        regs = '_'.join(ncfile.split('.')[0].split('_')[3: -1])
        rds = xr.load_dataset(path / ncfile)
    else:
        species = ''
        regs = ''
    rds = rds[feature].sel(level=level, method='nearest')
    proj = ccrs.PlateCarree(central_longitude=0)
    gl_list = []
    fg = rds.plot.contourf(col='regressors', add_colorbar=False,
                           col_wrap=col_wrap,
                           cmap=predict_cmap, center=0.0, extend=None,
                           levels=41, subplot_kws=dict(projection=proj),
                           transform=ccrs.PlateCarree(), figsize=figsize)
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
    gl_list[2].ylabels_left = False
    try:
        gl_list[3].ylabels_right = False
    except IndexError:
        pass
    fg.fig.tight_layout()
    fg.fig.subplots_adjust(right=0.96, left=0.04, wspace=0.15)
#    print('Caption: ')
#    print('The beta coeffciants for the water vapor anomalies MLR analysis in the 82 hPa level at 2004 to 2018')
    filename = 'MLR_{}_{}_map_{}_{}.png'.format(species, feature, level, regs)
    plt.savefig(savefig_path / filename, bbox_inches='tight')
    return fg


def plot_figure_13(path=work_chaim):
    ncfile = 'MLR_H2O_latlon_cdas-plags_ch4_enso_2004-2019.nc'
    fg = plot_feature_map(ncfile, path=path, feature='params', level=82)
    return fg


#def plot_figure_13(path=work_chaim, rds=None, save=True):
#    """params map (lat-lon) for cdas-plags, enso, ch4"""
#    import xarray as xr
#    import cartopy.crs as ccrs
#    from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
#    if rds is None:
#        rds = xr.open_dataset(
#            path /
#            'MLR_H2O_latlon_cdas-plags_ch4_enso_2004-2019.nc')
#    rds = rds['params'].sel(level=82, method='nearest')
#    proj = ccrs.PlateCarree(central_longitude=0)
##    fig, axes = plt.subplots(1, 3, figsize=(17, 3.0),
##                             subplot_kw=dict(projection=proj))
#    gl_list = []
#    fg = rds.plot.contourf(col='regressors', add_colorbar=False,
#                           cmap=predict_cmap, center=0.0, extend=None,
#                           levels=41, subplot_kws=dict(projection=proj),
#                           transform=ccrs.PlateCarree(), figsize=(17, 3))
#    cbar_kws = {'label': '', 'format': '%0.2f'}
#    cbar_ax = fg.fig.add_axes([0.1, 0.1, .8, .035])  # last num controls width
#    fg.add_colorbar(cax=cbar_ax, orientation="horizontal", **cbar_kws)
#    for ax in fg.axes.flatten():
#        ax.coastlines()
#        gl = ax.gridlines(
#            crs=ccrs.PlateCarree(),
#            linewidth=1,
#            color='black',
#            alpha=0.5,
#            linestyle='--',
#            draw_labels=True)
#        gl.xlabels_top = False
#        gl.xlabel_style = {'size': 9}
#        gl.ylabel_style = {'size': 9}
#        gl.xlines = True
#        gl.xlocator = mticker.FixedLocator([-180, -120, -60, 0, 60, 120, 180])
#        gl.ylocator = mticker.FixedLocator([-45, -30, -15, 0, 15, 30, 45])
#        gl.xformatter = LONGITUDE_FORMATTER
#        gl.yformatter = LATITUDE_FORMATTER
#        gl_list.append(gl)
#        ax = remove_regressors_and_set_title(ax)
#    gl_list[0].ylabels_right = False
#    gl_list[2].ylabels_left = False
#    fg.fig.tight_layout()
#    fg.fig.subplots_adjust(right=0.96, left=0.04, wspace=0.15)
#    print('Caption: ')
#    print('The beta coeffciants for the water vapor anomalies MLR analysis in the 82 hPa level at 2004 to 2018')
#    filename = 'MLR_H2O_params_map_82_cdas-plags_ch4_enso.png'
#    if save:
#        plt.savefig(savefig_path / filename, bbox_inches='tight')
#    return fg


def plot_figure_seasons_map(path=work_chaim, rds=None, field='r2_adj', level=82,
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
    fg = add_horizontal_colorbar(fg, rect=[0.1, 0.1, 0.8, 0.025],
                                 cbar_kwargs_dict=None)
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


def plot_figure_response_predict_maps(path=work_chaim, ncfile=None,
                                      species='H2O',
                                      field='response', bust='2010D-2011JFM',
                                      time_mean=None, time=None, 
                                      proj_key='PlateCarree',
                                      plot_kwargs=None, save=True):
    """response/predict maps (lat-lon) for cdas-plags, enso, ch4 for 2010D-2011JFM bust"""
    import xarray as xr
    import cartopy.crs as ccrs
    from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
    from ML_OOP_stratosphere_gases import plot_like_results
    time_dict = {'2010D-2011JFM': ['2010-12', '2011-03'],
                 '2015OND': ['2015-10', '2015-12'],
                 '2016OND': ['2016-10', '2016-12']}
    col_dict = {'response': 'regressors', 'predict': 'opr'}
    if species == 'H2O':
        if ncfile is None:
            ncfile = 'MLR_H2O_latlon_cdas-plags_ch4_enso_2004-2019.nc'
        rds = xr.open_dataset(path / ncfile)
        level = 82
        unit = 'ppmv'
    elif species == 't':
        if ncfile is None:
            ncfile = 'MLR_t_85hpa_latlon_cdas-plags_ch4_enso_1984-2019.nc'
        rds = xr.open_dataset(path / ncfile)
        level = 85
        unit = 'K'
    elif species == 'u':
        if ncfile is None:
            ncfile = 'MLR_u_85hpa_latlon_cdas-plags_ch4_enso_1984-2019.nc'
        rds = xr.open_dataset(path / ncfile)
        level = 85
        unit = r'm$\cdot$sec$^{-1}$'
    if time is None:
        time = time_dict.get(bust)
    fg = plot_like_results(rds, plot_key='{}_map'.format(field), level=level,
                           cartopy=True, time=time, time_mean=time_mean)
    rds = fg.data.sel(lat=slice(-60, 60))
    if time_mean == 'season':
        size = 4
    elif time_mean:
        size = 3
    else:
        size = rds.time.size
    if size == 4:
        figsize = (15, 7)
        s_adjust = dict(
            top=0.955,
            bottom=0.115,
            left=0.03,
            right=0.97,
            hspace=0.08,
            wspace=0.15)
        rect = [0.1, 0.06, 0.8, 0.01]
        i_label = 9
    elif size == 3:
        figsize = (15, 5.5)   # bottom=0.2, top=0.9, left=0.05
        s_adjust = dict(top=0.945,
                        bottom=0.125,
                        left=0.04,
                        right=0.97,
                        hspace=0.03,
                        wspace=0.18)
        rect = [0.1, 0.1, 0.8, 0.015]
        i_label = 6
    proj = getattr(ccrs, proj_key)(central_longitude=0.0)
    if time_mean is not None:
        if time_mean == 'season':
            row = 'season'
        elif time_mean:
            row = None
    else:
        row = 'time'
    plt_kwargs = {'add_colorbar': False, 'col_wrap': 3, 'center': 0, 'extend': None,
                  'cmap': predict_cmap, 'levels': 41, 'figsize': figsize}
    if plot_kwargs is not None:
        plt_kwargs.update(plot_kwargs)
    fg = rds.plot.contourf(col=col_dict.get(field), row=row,
                           subplot_kws={'projection': proj},
                           transform=ccrs.PlateCarree(),
                           **plt_kwargs)
    fg = add_horizontal_colorbar(
        fg, rect=rect, cbar_kwargs_dict={
            'label': unit})
    extent=[-170, 170, -57.5, 57.5]
    for i, ax in enumerate(fg.axes.flatten()):
        ax.coastlines(resolution='110m')
        if proj_key == 'PlateCarree':
            ax.set_extent(extent, crs=ccrs.PlateCarree())
            gl = ax.gridlines(crs=ccrs.PlateCarree(),
                              linewidth=1,
                              color='black',
                              alpha=0.5,
                              linestyle='--',
                              draw_labels=True)
            gl.xlabels_top = False
            gl.ylabels_right = False
            # if i < i_label:
            #     gl.xlabels_bottom = False
            gl.xlabel_style = {'size': 9}
            gl.ylabel_style = {'size': 9}
            gl.xlines = True
            gl.ylines = True
            gl.xlocator = mticker.FixedLocator([-180, -120, -60, 0, 60, 120, 180])
            gl.ylocator = mticker.FixedLocator([-45, -30, -15, 0, 15, 30, 45])
            gl.xformatter = LONGITUDE_FORMATTER
            gl.yformatter = LATITUDE_FORMATTER
        else:
            gl = ax.gridlines(crs=ccrs.PlateCarree(),
                              linewidth=1,
                              color='black',
                              alpha=0.5,
                              linestyle='--',
                              draw_labels=False)
        if field == 'response':
            ax = remove_regressors_and_set_title(ax)
        elif field == 'predict':
            ax = remove_anomaly_and_set_title(ax, species=species)
        if time_mean != 'season':
            ax = remove_time_and_set_date(ax)
    fg.fig.tight_layout()
    fg.fig.subplots_adjust(**s_adjust)
    if time is not None and time_mean == 'season':
        bust = '{}_seasons'.format(time)
    regs = '_'.join(ncfile.split('.')[0].split('_')[3: -1])
    filename = 'MLR_{}_{}_map_{}_{}_{}.png'.format(species, field, level, regs,
                                                   bust)
    if save:
        fg.fig.savefig(savefig_path / filename, bbox_inches='tight', orientation='landscape')
    return fg


def plot_figure_14(path=work_chaim):
    fg = plot_figure_response_predict_maps(path, species='H2O',
                                           field='response',
                                           bust='2010D-2011JFM', save=True)
    print('Caption: ')
    print('The water vapor anomalies predictor response map for the 82.54 hPa level in the 2010-D to 2011-JFM forecast bust.')
    return fg


def plot_figure_15(path=work_chaim):
    fg = plot_figure_response_predict_maps(path, species='t', field='response',
                                           bust='2010D-2011JFM', save=True)
    print('Caption: ')
    print('The air temperature anomalies predictor response map for the 85 hPa level in the 2010-D to 2011-JFM forecast bust.')
    return fg


def plot_figure_16(path=work_chaim):
    fg = plot_figure_response_predict_maps(path, species='H2O',
                                           field='predict',
                                           bust='2010D-2011JFM', save=True)
    print('Caption: ')
    print('The water vapor anomalies, reconstruction and residuals maps for the 82.54 hPa level in the 2010-D to 2011-JFM forecast bust.')
    return fg


def plot_figure_17(path=work_chaim):
    fg = plot_figure_response_predict_maps(path, species='H2O',
                                           field='response',
                                           bust='2015OND', save=True)
    print('Caption: ')
    print('The water vapor anomalies predictor response map for the 82.54 hPa level in the 2015-OND forecast bust.')
    return fg


def plot_figure_17_1(path=work_chaim):
    fg = plot_figure_response_predict_maps(
        path,
        species='H2O',
        time_mean=True,
        ncfile='MLR_H2O_latlon_cdas-plags_ch4_enso_poly_2_no_qbo^2_no_ch4_extra_2004-2019.nc',
        field='response',
        bust='2015OND', plot_kwargs={'figsize': (13.5, 5.4)},
        save=True)
    print('Caption: ')
    print('The water vapor anomalies predictor response map for the 82.54 hPa level in the 2015-OND forecast bust.')
    return fg


def plot_figure_18(path=work_chaim):
    fg = plot_figure_response_predict_maps(path, species='t', field='response',
                                           bust='2015OND', save=True)
    print('Caption: ')
    print('The air temperature anomalies predictor response map for the 85 hPa level in the 2015-OND forecast bust.')
    return fg


def plot_figure_19(path=work_chaim):
    fg = plot_figure_response_predict_maps(path, species='H2O',
                                           field='predict',
                                           bust='2015OND', save=True)
    print('Caption: ')
    print('The water vapor anomalies, reconstruction and residuals maps for the 82.54 hPa level in the 2015-OND forecast bust.')
    return fg


def plot_figure_19_1(path=work_chaim):
    fg = plot_figure_response_predict_maps(
        path,
        species='H2O',
        time_mean=None,
        ncfile='MLR_H2O_latlon_cdas-plags_ch4_enso_poly_2_no_qbo^2_no_ch4_extra_2004-2019.nc',
        field='predict',
        bust='2015OND',
        save=True)
    print('Caption: ')
    print('The water vapor anomalies predictor response map for the 82.54 hPa level in the 2015-OND forecast bust.')
    return fg


def plot_figure_20(path=work_chaim):
    fg = plot_figure_response_predict_maps(path, species='H2O',
                                           field='response',
                                           bust='2016OND', save=True)
    print('Caption: ')
    print('The water vapor anomalies predictor response map for the 82.54 hPa level in the 2016-OND forecast bust.')
    return fg


def plot_figure_21(path=work_chaim):
    fg = plot_figure_response_predict_maps(path, species='t', field='response',
                                           bust='2016OND', save=True)
    print('Caption: ')
    print('The air temperature anomalies predictor response map for the 85 hPa level in the 2016-OND forecast bust.')
    return fg


def plot_figure_22(path=work_chaim):
    fg = plot_figure_response_predict_maps(path, species='H2O',
                                           field='predict',
                                           bust='2016OND', save=True)
    print('Caption: ')
    print('The water vapor anomalies, reconstruction and residuals maps for the 82.54 hPa level in the 2016-OND forecast bust.')
    return fg


def plot_figure_23(path=work_chaim):
    fg = plot_figure_response_predict_maps(time=2009, species='H2O',
                                           field='response',
                                           time_mean='season', save=True)
    print('Caption: ')
    print('Seasonal water vapor anomalies predictor response maps for the 82.54 hPa level in 2009')
    return fg


def plot_figure_24(path=work_chaim):
    fg = plot_figure_response_predict_maps(time=2010, species='H2O',
                                           field='response',
                                           time_mean='season', save=True)
    print('Caption: ')
    print('Seasonal water vapor anomalies predictor response maps for the 82.54 hPa level in 2010')
    return fg


def plot_figure_25(path=work_chaim):
    fg = plot_figure_response_predict_maps(time=2009, species='t',
                                           field='response',
                                           time_mean='season', save=True)
    print('Caption: ')
    print('Seasonal air temperature anomalies predictor response maps for the 85 hPa level in 2009')
    return fg


def plot_figure_26(path=work_chaim):
    fg = plot_figure_response_predict_maps(time=2010, species='t',
                                           field='response',
                                           time_mean='season', save=True)
    print('Caption: ')
    print('Seasonal air temperature anomalies predictor response maps for the 85 hPa level in 2010')
    return fg


def plot_figure_27(path=work_chaim):
    fg = plot_figure_response_predict_maps(time=2009, species='u',
                                           field='response',
                                           time_mean='season', save=True)
    print('Caption: ')
    print('Seasonal zonal wind anomalies predictor response maps for the 85 hPa level in 2009')
    return fg


def plot_figure_28(path=work_chaim):
    fg = plot_figure_response_predict_maps(time=2010, species='u',
                                           field='response',
                                           time_mean='season', save=True)
    print('Caption: ')
    print('Seasonal zonal wind anomalies predictor response maps for the 85 hPa level in 2010')
    return fg


def plot_figure_poly2_lat(path=work_chaim):
    ncfile = 'MLR_H2O_latlon_cdas-plags_ch4_enso_poly_2_no_qbo^2_no_ch4_extra_2004-2019.nc'
    fg = plot_latlon_predict(ncfile, path=path, geo='lat', level=82.54,
                             bust_lines=True, save=True)


def plot_figure_poly2_lon(path=work_chaim):
    ncfile = 'MLR_H2O_latlon_cdas-plags_ch4_enso_poly_2_no_qbo^2_no_ch4_extra_2004-2019.nc'
    fg = plot_latlon_predict(ncfile, path=path, geo='lon', level=82.54,
                             bust_lines=True, save=True)
    
def plot_figure_poly2_params(path=work_chaim):
    ncfile = 'MLR_H2O_latlon_cdas-plags_ch4_enso_poly_2_no_qbo^2_no_ch4_extra_2004-2019.nc'
    fg = plot_feature_map(
        ncfile,
        path=path,
        feature='params',
        level=82,
        col_wrap=3,
        figsize=(
            15,
            5))
    plt.subplots_adjust(top=1.0,
                        bottom=0.101,
                        left=0.051,
                        right=0.955,
                        hspace=0.0,
                        wspace=0.21)
    return fg


def plot_1984_2004_comparison(path=work_chaim, reg_drop=None):
    import xarray as xr

    def drop_reg_from_da(da, to_drop=None):
        if to_drop is not None:
            ds = da.to_dataset('regressors')
            ds = ds[[x for x in ds.data_vars if to_drop not in x]]
            da = ds.to_array('regressors')
            return da
    rds_1984 = xr.open_dataset(
        path /
        'MLR_H2O_latpress_cdas-plags_ch4_enso_1984-2019.nc')
    rds_1984_82 = rds_1984.sel(level=82, method='nearest')
    predict_1984 = rds_1984_82['predict']
    X = rds_1984_82['X']
    rds_2004 = xr.open_dataset(
        path /
        'MLR_H2O_latlon_cdas-plags_ch4_enso_2004-2019.nc')
    rds_2004_82 = rds_2004.sel(
        level=82,
        method='nearest').mean(
        'lon',
        keep_attrs=True)
    params_2004 = rds_2004_82['params'].interp(lat=rds_1984['lat'])
    if reg_drop is not None:
        params_2004 = drop_reg_from_da(params_2004, reg_drop)
        X = drop_reg_from_da(X, reg_drop)
    intercept_2004 = rds_2004_82['intercept'].interp(lat=rds_1984['lat'])
    predict_2004 = params_2004.dot(X, dims=['regressors']) + intercept_2004
    compare = xr.concat([predict_1984, predict_2004], 'beta')
    compare['beta'] = ['1984 coeffs', '2004 coeffs']
    compare = compare.T
    compare.plot.contourf(levels=41, figsize=(15, 6), row='beta')
    plt.subplots_adjust(top=0.934,
                        bottom=0.09,
                        left=0.034,
                        right=0.84,
                        hspace=0.185,
                        wspace=0.2)
    return compare


def process_qbo_enso_from_2004_beta(path=work_chaim):
    import xarray as xr
    ncfile = 'MLR_H2O_latpress_cdas-plags_enso_1984-2019_from_2004_beta.nc'
    if (path/ncfile).is_file():
        print('plotting..')
        fg = plot_latlon_predict(ncfile, path=work_chaim, geo='lat',
                                 level=82.54, st_year=1984,
                                 bust_lines=True, save=True)
        return fg
    else:
        print('proccesing...')
        rds_all = xr.open_dataset(path / 'MLR_H2O_latpress_cdas-plags_enso_poly_2_no_qbo^2_1984-2019.nc')
        rds_2004 = xr.open_dataset(
                path /
                'MLR_H2O_latpress_cdas-plags_enso_poly_2_no_qbo^2_2004-2019.nc')
        rds_2004 = rds_2004.drop(['X', 'original', 'predict', 'resid'])
        rds_2004['time'] = rds_all['time']
        rds_2004['predict'] = rds_2004['params'].sortby('regressors').dot(
            rds_all['X'].sortby('regressors'), dims=['regressors']) + rds_2004['intercept']
        rds_2004['original'] = rds_all['original']
        rds_2004['resid'] = rds_2004['original'] - rds_2004['predict']
        rds_2004.to_netcdf(path / ncfile)
        print('file saved, run again to plot.')
        return rds_2004


def plot_1984_2004_prediction_vise_versa(path=work_chaim,
                                         reg='poly_2_no_qbo^2'):
    import xarray as xr
    # first open 1984 and 2004 analysis for enso and qbo only:
    if reg is not None:
        rds_1984 = xr.open_dataset(
            path /
            'MLR_H2O_latpress_cdas-plags_enso_{}_1984-2004.nc'.format(reg))
    else:
        rds_1984 = xr.open_dataset(
            path /
            'MLR_H2O_latpress_cdas-plags_enso_1984-2004.nc')
    rds_1984_82 = rds_1984.sel(level=82, method='nearest')
    if reg is not None:
        rds_2004 = xr.open_dataset(
            path /
            'MLR_H2O_latpress_cdas-plags_enso_{}_2004-2019.nc'.format(reg))
    else:
        rds_2004 = xr.open_dataset(
            path /
            'MLR_H2O_latpress_cdas-plags_enso_2004-2019.nc')
    rds_2004_82 = rds_2004.sel(level=82, method='nearest')
    # now predict 2004-2019 using 1984 fit:
    predict_2004_from_1984 = rds_1984_82['params'].dot(
        rds_2004_82['X'], dims=['regressors']) + rds_1984_82['intercept']
    compare_2004_from_1984 = xr.concat(
        [rds_1984_82['original'], predict_2004_from_1984], 'time')
    # now predict 1984-2004 using 2004 fit:
    predict_1984_from_2004 = rds_2004_82['params'].dot(
        rds_1984_82['X'], dims=['regressors']) + rds_2004_82['intercept']
    compare_1984_from_2004 = xr.concat(
        [rds_2004_82['original'], predict_1984_from_2004], 'time')
    compare = xr.concat(
        [compare_2004_from_1984, compare_1984_from_2004], 'compare')
    compare['compare'] = [r'2004-2019 data from 1984 $\beta$',
                          r'1984-2004 data from 2004 $\beta$']
    compare = compare.sortby('time').transpose('lat', ...)
    fg = compare.plot.contourf(levels=41, row='compare', figsize=(20, 6))
    axes = fg.axes.flatten()
    axes[0].set_title(r'2004-2019 prediction from 1984-2004 $\beta$')
    axes[1].set_title(r'1984-2004 prediction from 2004-2019 $\beta$')
    plt.draw()
    return compare


def plot_enso_events(path=work_chaim, year='1984'):
    import xarray as xr
    enso = xr.load_dataset(path/'MLR_H2O_latpress_cdas-plags_enso_{}-2019_neutral_enso.nc'.format(year))
    la_nina = xr.load_dataset(path/'MLR_H2O_latpress_cdas-plags_enso_{}-2019_la_nina.nc'.format(year))
    el_nino = xr.load_dataset(path/'MLR_H2O_latpress_cdas-plags_enso_{}-2019_el_nino.nc'.format(year))
    la_nina_size = la_nina['original'].dropna('time').time.size
    el_nino_size = el_nino['original'].dropna('time').time.size
    enso_size = enso['original'].dropna('time').time.size
    compare = xr.concat([enso['params'], la_nina['params'], el_nino['params']], 'events')
    compare['events'] = [
        'neutral enso ({})'.format(enso_size),
        'la nina ({})'.format(la_nina_size),
        'el nino ({})'.format(el_nino_size)]
    fg = compare.plot.contourf(
        yscale='log',
        row='regressors',
        col='events',
        levels=41, yincrease=False,
        figsize=(
            20,
            10))
    [ax.invert_yaxis() for ax in fg.axes.flat]
    [ax.yaxis.set_major_formatter(ScalarFormatter()) for ax in
     fg.axes.flat]
    fg.fig.suptitle('From MLR on {}-2019 data'.format(year))
    return fg


def plot_enso_scatter_era5(path=work_chaim, qbo_lag=2, plot=True):
    """regress out CH4 and qbo_cdas at lag 2/3? from ERA5-q,
    level 82.54 hPa and scatter plot with ENSO seasons/annual"""
    from make_regressors import load_all_regressors
    from aux_functions_strat import normalize_xr
    from aux_functions_strat import lat_mean
    from sklearn.linear_model import LinearRegression
    from aux_functions_strat import deseason_xr
    from aux_functions_strat import dim_intersection
    from make_regressors import create_season_avg_nino
    import xarray as xr
    # load era5_q:
    era5_q = xr.load_dataset(path/'era5_wv_anoms_82hPa.nc')
    era5_q = era5_q.mean('lon')
    era5_q = era5_q.sel(lat=slice(-15, 15))
    era5_q = lat_mean(era5_q).q
    # load regressors:
    ds = load_all_regressors()
    # regress out ch4:
    ch4 = normalize_xr(ds['ch4'].dropna('time'), norm=5)
#    qbos = deseason_xr(qbos.to_array('regressors'), how='mean').to_dataset('regressors')
#    radio = radio.sel(time=new_time)
    lr = LinearRegression()
#    X = qbos.to_array('regressors').T
    X = ch4.values.reshape(-1,1)
    lr.fit(X, era5_q.squeeze())
    era5_q = era5_q - lr.predict(X)
    # regress out qbo at lag = qbo_lag:
    qbo = deseason_xr(ds['qbo_cdas'].dropna('time'), how='mean')
    qbo = qbo.shift(time=qbo_lag)
    new_time = dim_intersection([qbo, era5_q], 'time')
    qbo = qbo.sel(time=new_time)
    era5_q = era5_q.sel(time=new_time)
    lr = LinearRegression()
    X = qbo.values.reshape(-1, 1)
    lr.fit(X, era5_q.squeeze())
    era5_q = era5_q - lr.predict(X)
    # scatter plot with enso:
    enso = ds['anom_nino3p4'].dropna('time')
    enso = create_season_avg_nino()
    enso = enso.sel(time=new_time)
    if plot:
        fig, axes = plt.subplots(2,2, sharex=True, sharey=True, figsize=(10, 8))
        seasons = ['JJA', 'MAM', 'SON', 'DJF']
        fig.suptitle('ERA5 specific humidity (1979-2019) at 82.54 hPa (CH4 and QBO_CDAS (lag-2) regressed out) vs. ENSO')
        for i, ax in enumerate(axes.flatten()):
            x = enso.sel(time=enso['time.season']==seasons[i])
            y = era5_q.sel(time=era5_q['time.season']==seasons[i])
            ax.scatter(x=x, y=y)
            ax.set_title(seasons[i])
            ax.grid()
            ax.set_ylabel('ERA5 q anomalies [ppmv]')
            ax.set_xlabel('ENSO 3.4')
        for ax in fig.get_axes():
            ax.label_outer()
    return fig
# Check Diallo 2018 et al:
# 1)do lms with ENSO, QBO, AOD (or vol or aot)
# 2)do run_ML with them (lat-pressure only)
# 3) do 1) +2) with ENSO, AOD only
# subtract 2) from 3) and compare with QBO*beta(QBO)