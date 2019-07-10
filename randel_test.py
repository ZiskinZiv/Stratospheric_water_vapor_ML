#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 13:12:22 2019

@author: shlomi
"""
from strat_startup import *
sound_path = work_chaim / 'sounding'
def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=["black", "white"],
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A list or array of two color specifications.  The first is used for
        values below a threshold, the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """
    import matplotlib
    import numpy as np
    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts

def get_randel_corr():
    import numpy as np
    import xarray as xr
    import aux_functions_strat as aux
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    from strato_soundings import calc_cold_point_from_sounding
    radio_cold = calc_cold_point_from_sounding(path=sound_path, times=('1993', '2017'),
                                  plot=False, return_cold=True)
    radio_cold2 = calc_cold_point_from_sounding(path=sound_path, times=('1993', '2017'),
                                  plot=False, return_cold=True,
                                  stations=['BRM00082332','RMM00091376'])
    radio_cold2.name = 'radiosonde_cold_point_anomalies_2_stations'
    swoosh = xr.open_dataset(work_chaim / 'swoosh_latpress-2.5deg.nc')
    haloe = xr.open_dataset(work_chaim /
                            'swoosh-v02.6-198401-201812/swoosh-v02.6-198401-201812-latpress-2.5deg-L31.nc', decode_times=False)
    haloe['time'] = swoosh.time
    haloe_names = [x for x in haloe.data_vars.keys()
                   if 'haloe' in x and 'h2o' in x]
    haloe = haloe[haloe_names].sel(level=slice(83, 81), lat=slice(-20, 20))
    # com=swoosh.combinedanomfillanomh2oq
    com = swoosh.combinedanomfillanomh2oq
    com_nofill = swoosh.combinedanomh2oq
    com = com.sel(level=slice(83, 81), lat=slice(-20, 20))
    com_nofill = com_nofill.sel(level=slice(83, 81), lat=slice(-20, 20))
    weights = np.cos(np.deg2rad(com['lat'].values))
    com_latmean = (weights * com).sum('lat') / sum(weights)
    com_latmean_2M_lagged = com_latmean.shift(time=-2)
    com_latmean_2M_lagged.name = com_latmean.name + ' + 2M lag'
    com_nofill_latmean = (weights * com_nofill).sum('lat') / sum(weights)
    haloe_latmean = (weights * haloe.haloeanomh2oq).sum('lat') / sum(weights)
    era40 = xr.open_dataarray(work_chaim / 'ERA40_T_mm_eq.nc')
    era40 = era40.sel(level=100)
    weights = np.cos(np.deg2rad(era40['lat'].values))
    era40_latmean = (weights * era40).sum('lat') / sum(weights)
    era40anom_latmean = aux.deseason_xr(era40_latmean)
    era40anom_latmean.name = 'era40_100hpa_anomalies'
    era5 = xr.open_dataarray(work_chaim / 'ERA5_T_eq_all.nc')
    cold_point = era5.sel(level=slice(150, 50)).min(['level', 'lat',
                                                     'lon'])
    cold_point = aux.deseason_xr(cold_point)
    cold_point.name = 'cold_point_from_era5'
    era5 = era5.mean('lon').sel(level=100)
    weights = np.cos(np.deg2rad(era5['lat'].values))
    era5_latmean = (weights * era5).sum('lat') / sum(weights)
    era5anom_latmean = aux.deseason_xr(era5_latmean)
    era5anom_latmean.name = 'era5_100hpa_anomalies'
    merra = xr.open_dataarray(work_chaim / 'T_regrided.nc')
    merra['time'] = pd.date_range(start='1979', periods=merra.time.size, freq='MS')
    merra = merra.mean('lon').sel(lat=slice(-20, 20), level=100)
    weights = np.cos(np.deg2rad(merra['lat'].values))
    merra_latmean = (weights * merra).sum('lat') / sum(weights)
    merra_latmean.name = 'merra'
    merraanom_latmean = aux.deseason_xr(merra_latmean)
    merraanom_latmean.name = 'merra_100hpa_anomalies'
    to_compare = xr.merge([com_nofill_latmean.squeeze(drop=True),
                           com_latmean.squeeze(drop=True),
                           com_latmean_2M_lagged.squeeze(drop=True),
                           cold_point,
                           era40anom_latmean.squeeze(drop=True),
                           era5anom_latmean.squeeze(drop=True),
                           merraanom_latmean.squeeze(drop=True),
                           radio_cold.squeeze(drop=True),
                           radio_cold2.squeeze(drop=True)])
    # to_compare.to_dataframe().plot()
    # plt.figure()
    # sns.heatmap(to_compare.to_dataframe().corr(), annot=True)
    # plt.subplots_adjust(left=0.35, bottom=0.4, right=0.95)
    to_compare.sel(time=slice('1993', '2018')).to_dataframe().plot()
    corr = to_compare.to_dataframe().corr()
    # sns.heatmap(to_compare.sel(time=slice('1993', '2018')).to_dataframe().corr(),
    #             annot=True)
    fig, ax = plt.subplots(figsize=(11, 11))

    im, cbar = heatmap(corr.values, corr.index.values, corr.columns, ax=ax,
                       cmap="YlGn", cbarlabel="correlation")
    texts = annotate_heatmap(im, valfmt="{x:.2f}")
    fig.tight_layout()
    plt.show()
    # plt.subplots_adjust(left=0.35, bottom=0.4, right=0.95)
    return to_compare


def proc_coldest_point(work_path):
    """create coldest point index by using era5 4xdaily data"""
    import xarray as xr
    from aux_functions_strat import lat_mean, deseason_xr
    T_lat_lon_4Xdaily = xr.open_dataset(work_path / 'cold_point_era5.nc')
    T_daily_zonal = T_lat_lon_4Xdaily.resample(time='1D').mean('time').mean('lon')
    T_daily = lat_mean(T_daily_zonal)
    # 1)open mfdataset the temperature data
    # 2)selecting -15 to 15 lat, and maybe the three levels (125,100,85)
    # 2a) alternativly, find the 3 lowest temperature for each lat/lon in 
    # the level dimension
    # 3) run a quadratic fit to the three points coldest points and select
    # the minimum, put it in the lat/lon grid.
    # 4) resample to monthly means and average over lat/lon and voila!
    return da
    
