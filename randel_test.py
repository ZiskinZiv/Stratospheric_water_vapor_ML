#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 13:12:22 2019

@author: shlomi
"""
from strat_paths import work_chaim
sound_path = work_chaim / 'sounding'
sean_tropopause_path = work_chaim / 'Sean - tropopause'


def read_ascii_randel(path, filename='h2o_all_timeseries_for_corr.dat'):
    import pandas as pd
    import numpy as np
    with open(path / filename) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    content = [x.split() for x in content]
    content.pop(0)
    # flatten list:
    flat_content = [item for sublist in content for item in sublist]
    # turn it to float:
    flat_content = [float(x) for x in flat_content]
    # find first bad value:
    pos = [i for i, x in enumerate(flat_content) if x == 1e36][0]
    # seperate to two list:
    dates = pd.to_datetime(flat_content[0:pos-1], origin='julian', unit='D')
    start_date = str(dates.year[0]) + '-' + str(dates.month[0]) + '-' + '01'
    dates_new = pd.date_range(start_date, freq='MS', periods=len(dates))
    wv = flat_content[pos: -1]
    df = pd.DataFrame(wv, index=dates_new)
    df = df.replace(1e36, np.nan)
    df.index.name = 'time'
    df.columns = ['wv_anoms_HALOE_MLS']
    ds = df.to_xarray()
    da = ds.to_array(name='wv_anoms_HALOE_MLS').squeeze(drop=True)
    return da


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


def load_cpt_models(lats=[-15, 15], plot=False):
    """load cold_point temperature from various models"""
    import xarray as xr
    import aux_functions_strat as aux
    import pandas as pd
    import matplotlib.pyplot as plt

    def produce_anoms_from_model(path_and_filename, cpt='ctpt', lats=lats,
                                 time_dim='time'):
        if 'cfsr' in path_and_filename.as_posix():
            ds = xr.open_dataset(path_and_filename, decode_times=False)
            ds[time_dim] = pd.to_datetime(
                ds[time_dim], origin='julian', unit='D')
        else:
            ds = xr.open_dataset(path_and_filename)
        ds = ds.sortby('lat')
        da = ds[cpt].sel(lat=slice(lats[0], lats[1])).mean('lat')
        da_anoms = aux.deseason_xr(da, how='mean')
        da_anoms = add_times_to_attrs(da_anoms)
        # replace time with monthly means that start with 1-1 of each month:
        start_date = pd.to_datetime(
            da_anoms[time_dim][0].values).strftime('%Y-%m')
        new_time = pd.date_range(
            start_date,
            periods=da_anoms[time_dim].size,
            freq='MS')
        da_anoms[time_dim] = new_time
        return da_anoms
    # era40:
    era40_anoms = produce_anoms_from_model(
        sean_tropopause_path / 'era40.tp.monmean.zm.nc')
    era40_anoms.name = 'era40_cpt_anoms_eq'
    # era interim:
    erai_anoms = produce_anoms_from_model(
        sean_tropopause_path / 'erai.tp.monmean.zm.nc')
    erai_anoms.name = 'era_interim_cpt_anoms_eq'
    # jra25:
    jra25_anoms = produce_anoms_from_model(
        sean_tropopause_path / 'jra25.tp.monmean.zm.nc')
    jra25_anoms.name = 'jra25_cpt_anoms_eq'
    # jra55:
    jra55_anoms = produce_anoms_from_model(
        sean_tropopause_path / 'jra55.monmean.zm.nc')
    jra55_anoms.name = 'jra55_cpt_anoms_eq'
    # merra:
    merra_anoms = produce_anoms_from_model(
        sean_tropopause_path / 'merra.tp.monmean.zm.nc')
    merra_anoms.name = 'merra_cpt_anoms_eq'
    # merra2:
    merra2_anoms = produce_anoms_from_model(
        sean_tropopause_path / 'merra2.tp.monmean.zm.nc')
    merra2_anoms.name = 'merra2_cpt_anoms_eq'
    # ncep:
    ncep_anoms = produce_anoms_from_model(
        sean_tropopause_path / 'ncep.tp.monmean.zm.nc')
    ncep_anoms.name = 'ncep_cpt_anoms_eq'
    # cfsr:
    cfsr_anoms = produce_anoms_from_model(
        sean_tropopause_path / 'cfsr.monmean.zm.nc')
    cfsr_anoms.name = 'cfsr_cpt_anoms_eq'
    # merge all:
    cpt_models = xr.merge([era40_anoms,
                           erai_anoms,
                           jra25_anoms,
                           jra55_anoms,
                           merra_anoms,
                           merra2_anoms,
                           ncep_anoms,
                           cfsr_anoms])
    if plot:
        # fig, ax = plt.subplots(figsize=(11, 11), sharex=True)
        df = cpt_models.to_dataframe()
        model_names = [x.split('_')[0] for x in df.columns]
        df = df[df.index > '1979']
        for i, col in enumerate(df.columns):
            df.iloc[:, i] += i*5.0
        ax = df.plot(legend=False, figsize=(11, 11))
        ax.grid()
        ax.legend(model_names, loc='best', fancybox=True, framealpha=0.5) # , bbox_to_anchor=(0.85, 1.05))
        ax.set_title('Cold-Point-Temperature anoms from various models')
    return cpt_models


def add_times_to_attrs(da, time_dim='time', mm_only=True):
    import pandas as pd
    da_no_nans = da.dropna(time_dim)
    dt_min = da_no_nans.time.values[0]
    dt_max = da_no_nans.time.values[-1]
    if mm_only:
        dt_min_str = pd.to_datetime(dt_min).strftime('%Y-%m')
        dt_max_str = pd.to_datetime(dt_max).strftime('%Y-%m')
    else:
        dt_min_str = pd.to_datetime(dt_min).strftime('%Y-%m-%d')
        dt_max_str = pd.to_datetime(dt_max).strftime('%Y-%m-%d')
    da.attrs['first_date'] = dt_min_str
    da.attrs['last_date'] = dt_max_str
    return da


def load_wv_data(lag=2, plot=False):
    import xarray as xr
    import numpy as np
    import aux_functions_strat as aux

    # first load swoosh:
    swoosh = xr.open_dataset(work_chaim / 'swoosh_latpress-2.5deg.nc')
    com_nofill = swoosh.combinedanomh2oq
    com_nofill = com_nofill.sel(level=slice(83, 81)).squeeze(drop=True)
    weights = np.cos(np.deg2rad(com_nofill['lat']))
    swoosh_combined_near_global = (
        weights.sel(lat=slice(-60, 60)) * com_nofill.sel(lat=slice(-60, 60))).sum('lat') / sum(weights)
    swoosh_combined_equatorial = (
        weights.sel(lat=slice(-15, 15)) * com_nofill.sel(lat=slice(-15, 15))).sum('lat') / sum(weights)
    swoosh_anoms_near_global = aux.deseason_xr(
        swoosh_combined_near_global, how='mean')
    swoosh_anoms_near_global = add_times_to_attrs(swoosh_anoms_near_global)
    swoosh_anoms_near_global.name = 'swoosh_anoms_near_global'
    swoosh_anoms_equatorial = aux.deseason_xr(
        swoosh_combined_equatorial, how='mean')
    swoosh_anoms_equatorial.name = 'swoosh_anoms_equatorial'
    swoosh_anoms_equatorial = add_times_to_attrs(swoosh_anoms_equatorial)
    wv_anoms_randel = read_ascii_randel(cwd)
    wv_anoms_randel.name = 'wv_anoms_near_global_from_randel'
    wv_anoms_randel = add_times_to_attrs(wv_anoms_randel)
    wv_anoms = xr.merge([wv_anoms_randel,
                         swoosh_anoms_near_global,
                         swoosh_anoms_equatorial])
    print('loaded wv anoms data...')
    if plot:
        df = wv_anoms.to_dataframe()
        #model_names = ['Randel near global', 'SWOOSH near global',
        #               'SWOOSH equatorial']
        for i, col in enumerate(df.columns):
            df.iloc[:, i] += i*0.45
        ax = df.plot(legend=True, figsize=(17, 5))
        ax.grid()
        ax.grid('on', which='minor', axis='x' )
        # ax.legend(model_names, loc='best', fancybox=True, framealpha=0.5) # , bbox_to_anchor=(0.85, 1.05))
        ax.set_title('Water Vapor anoms from Randel and SWOOSH')
    # now add 2 month lag:
    if lag is not None:
        for da in wv_anoms.data_vars.values():
            new_da = da.shift(time=-1 * lag)
            new_da.name = da.name + '_' + str(lag) + 'm'
            wv_anoms[new_da.name] = new_da
        print('added {} month lag to anom data...'.format(lag))
    return wv_anoms


def gantt_chart(ds):
    import pandas as pd
    from matplotlib.pyplot import cm 
    import numpy as np
    import matplotlib.pyplot as plt
    from datetime import datetime, timedelta
    df = ds.to_dataframe()
    x2 = df.index[-1].to_pydatetime()
    x1 = df.index[0].to_pydatetime()
    y = df.index.astype(int)
    names = df.columns
    
    labs, tickloc, col = [], [], []
    
    # create color iterator for multi-color lines in gantt chart
    color=iter(cm.Dark2(np.linspace(0,1,len(y))))
    
    plt.figure(figsize=(8,10))
    fig, ax = plt.subplots()
    
    # generate a line and line properties for each station
    for i in range(len(y)):
        c=next(color)
        
        plt.hlines(i+1, x1[i], x2[i], label=y[i], color=c, linewidth=2)
        labs.append(names[i].title()+" ("+str(y[i])+")")
        tickloc.append(i+1)
        col.append(c)
    plt.ylim(0,len(y)+1)
    plt.yticks(tickloc, labs)
    
    # create custom x labels
    plt.xticks(np.arange(datetime(np.min(x1).year,1,1),np.max(x2)+timedelta(days=365.25),timedelta(days=365.25*5)),rotation=45)
    plt.xlim(datetime(np.min(x1).year,1,1),np.max(x2)+timedelta(days=365.25))
    plt.xlabel('Date')
    plt.ylabel('USGS Official Station Name and Station Id')
    plt.grid()
    plt.title('USGS Station Measurement Duration')
    # color y labels to match lines
    gytl = plt.gca().get_yticklabels()
    for i in range(len(gytl)):
        gytl[i].set_color(col[i])
    plt.tight_layout()
    return


def correlate_wv_models_radio(
        times1=['1993', '2017'], times2=['2005', '2017']):
    from strato_soundings import calc_cold_point_from_sounding
    from strato_soundings import get_cold_point_from_wang_sounding
    import xarray as xr
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    radio_cold3_t1 = calc_cold_point_from_sounding(path=sound_path,
                                                   times=(
                                                       times1[0], times1[1]),
                                                   plot=False, return_mean=True)
    radio_cold3_t2 = calc_cold_point_from_sounding(path=sound_path,
                                                   times=(
                                                          times2[0], times2[1]),
                                                          plot=False, return_mean=True)
    radio_cold3_t1.name = 'radio_cpt_anoms_3_stations_randel'
    radio_cold3_t2.name = 'radio_cpt_anoms_3_stations_randel'
    ds = get_cold_point_from_wang_sounding()
    radio_wang = ds.to_array().mean('variable')
    radio_wang.name = 'radio_cpt_anoms_3_stations_wang'
    wv_anoms = load_wv_data()
    cpt_models = load_cpt_models()
    to_compare1 = xr.merge([wv_anoms, cpt_models, radio_cold3_t1, radio_wang])
    to_compare2 = xr.merge([wv_anoms, cpt_models, radio_cold3_t2, radio_wang])
    to_compare1 = to_compare1.sel(time=slice(times1[0], times1[1]))
    to_compare2 = to_compare2.sel(time=slice(times2[0], times2[1]))
    corr1 = to_compare1.to_dataframe().corr()
    corr2 = to_compare2.to_dataframe().corr()
    mask = np.zeros_like(corr1)
    mask[np.triu_indices_from(mask)] = True
    df_mask = corr1.copy()
    df_mask[:] = mask
    df_mask = df_mask.astype(bool)
    corr1[df_mask] = corr2[df_mask]
    fig, ax = plt.subplots(figsize=(11, 11))
#    mask = np.zeros_like(corr)
#    mask[np.triu_indices_from(mask)] = True
    h = sns.heatmap(corr1, annot=True, cmap="YlGn", ax=ax,
                    cbar=True)
    h.set_xticklabels(
        h.get_xticklabels(),
        rotation=45,
        horizontalalignment='right')
#    im, cbar = heatmap(corr.values, corr.index.values, corr.columns, ax=ax,
#                       cmap="YlGn", cbarlabel="correlation")
#    texts = annotate_heatmap(im, valfmt="{x:.2f}")
    ax.hlines([6, 16], xmin=0, xmax=6, color='r')
    ax.vlines([0, 6], ymin=6, ymax=16, color='r')
    font = {'family': 'serif',
            'color': 'darkred',
            'weight': 'normal',
            'size': 16,
            }
#    for lab, annot in zip(ax.get_yticklabels(), ax.texts):
#        text =  lab.get_text()
#        if text == 'radio_cpt_anoms_3_stations_randel': # lets highlight row 2
#            # set the properties of the ticklabel
#            # lab.set_weight('bold')
#            # lab.set_size(20)
#            lab.set_color('purple')
#            # set the properties of the heatmap annot
#            annot.set_weight('bold')
#            annot.set_color('purple')
#            # annot.set_size(20)
    ax.set_title('Upper right heatmap: {} to {}, lower left heatmap: {} to {}.'.format(
                times2[0], times2[1], times1[0], times1[1]), fontdict=font)
#    fig.tight_layout()
#    fig.colorbar(h.get_children()[0], ax=axes[1])
    plt.subplots_adjust(left=0.3, bottom=0.25, right=0.95)
    # plt.tight_layout()
    plt.show()
    # plt.subplots_adjust(left=0.35, bottom=0.4, right=0.95)
    return fig


def get_randel_corr(lats=[-10, 10], times=['1993', '2017']):
    import numpy as np
    import xarray as xr
    import aux_functions_strat as aux
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    from strato_soundings import calc_cold_point_from_sounding
#    radio_cold = calc_cold_point_from_sounding(path=sound_path, times=('1993', '2017'),
#                                  plot=False, return_cold=True)
    radio_cold3 = calc_cold_point_from_sounding(path=sound_path,
                                                times=(times[0], times[1]),
                                                plot=False, return_mean=True)
    radio_cold3.name = 'radiosonde_cold_point_anomalies_3_stations'
    radio_smooth = radio_cold3.rolling(time=3, center=True).mean()
    radio_smooth.name = 'radiosonde_smooth_3_months'
    wv_anoms_HM = read_ascii_randel(cwd)
    wv_anoms_HM_2M_lagged = wv_anoms_HM.shift(time=-2)
    wv_anoms_HM_2M_lagged.name = wv_anoms_HM.name + ' + 2M lag'
    swoosh = xr.open_dataset(work_chaim / 'swoosh_latpress-2.5deg.nc')
    haloe = xr.open_dataset(work_chaim /
                            'swoosh-v02.6-198401-201812/swoosh-v02.6-198401-201812-latpress-2.5deg-L31.nc', decode_times=False)
    haloe['time'] = swoosh.time
    haloe_names = [x for x in haloe.data_vars.keys()
                   if 'haloe' in x and 'h2o' in x]
    haloe = haloe[haloe_names].sel(level=slice(83, 81), lat=slice(lats[0],
                                                                  lats[1]))
    # com=swoosh.combinedanomfillanomh2oq
    com = swoosh.combinedanomfillanomh2oq
    com_nofill = swoosh.combinedanomh2oq
    com = com.sel(level=slice(83, 81), lat=slice(lats[0], lats[1]))
    com_nofill = com_nofill.sel(level=slice(83, 81), lat=slice(lats[0],
                                                               lats[1]))
    weights = np.cos(np.deg2rad(com['lat'].values))
    com_latmean = (weights * com).sum('lat') / sum(weights)
    com_latmean_2M_lagged = com_latmean.shift(time=-2)
    com_latmean_2M_lagged.name = com_latmean.name + ' + 2M lag'
    com_nofill_latmean = (weights * com_nofill).sum('lat') / sum(weights)
    com_nofill_latmean_2M_lagged = com_nofill_latmean.shift(time=-2)
    com_nofill_latmean_2M_lagged.name = com_nofill_latmean.name + ' + 2M lag'
    haloe_latmean = (weights * haloe.haloeanomh2oq).sum('lat') / sum(weights)
    era40 = xr.open_dataarray(work_chaim / 'ERA40_T_mm_eq.nc')
    era40 = era40.sel(level=100)
    weights = np.cos(np.deg2rad(era40['lat'].values))
    era40_latmean = (weights * era40).sum('lat') / sum(weights)
    era40anom_latmean = aux.deseason_xr(era40_latmean, how='mean')
    era40anom_latmean.name = 'era40_100hpa_anomalies'
    era5 = xr.open_dataarray(work_chaim / 'ERA5_T_eq_all.nc')
    cold_point = era5.sel(level=slice(150, 50)).min(['level', 'lat',
                                                     'lon'])
    cold_point = aux.deseason_xr(cold_point, how='mean')
    cold_point.name = 'cold_point_from_era5'
    era5 = era5.mean('lon').sel(level=100)
    weights = np.cos(np.deg2rad(era5['lat'].values))
    era5_latmean = (weights * era5).sum('lat') / sum(weights)
    era5anom_latmean = aux.deseason_xr(era5_latmean, how='mean')
    era5anom_latmean.name = 'era5_100hpa_anomalies'
    merra = xr.open_dataarray(work_chaim / 'T_regrided.nc')
    merra['time'] = pd.date_range(start='1979', periods=merra.time.size, freq='MS')
    merra = merra.mean('lon').sel(lat=slice(-20, 20), level=100)
    weights = np.cos(np.deg2rad(merra['lat'].values))
    merra_latmean = (weights * merra).sum('lat') / sum(weights)
    merra_latmean.name = 'merra'
    merraanom_latmean = aux.deseason_xr(merra_latmean, how='mean')
    merraanom_latmean.name = 'merra_100hpa_anomalies'
    to_compare = xr.merge([com_nofill_latmean.squeeze(drop=True),
                           com_nofill_latmean_2M_lagged.squeeze(drop=True),
                           com_latmean.squeeze(drop=True),
                           com_latmean_2M_lagged.squeeze(drop=True),
                           cold_point,
                           wv_anoms_HM, wv_anoms_HM_2M_lagged,
                           era40anom_latmean.squeeze(drop=True),
                           era5anom_latmean.squeeze(drop=True),
                           merraanom_latmean.squeeze(drop=True),
                           radio_cold3.squeeze(drop=True),
                           radio_smooth.squeeze(drop=True)])
                           # radio_cold2.squeeze(drop=True)])
    # to_compare.to_dataframe().plot()
    # plt.figure()
    # sns.heatmap(to_compare.to_dataframe().corr(), annot=True)
    # plt.subplots_adjust(left=0.35, bottom=0.4, right=0.95)
    # to_compare.sel(time=slice('1993', '2018')).to_dataframe().plot()
    to_compare = to_compare.sel(time=slice(times[0], times[1]))
    corr = to_compare.to_dataframe().corr()
    # sns.heatmap(to_compare.sel(time=slice('1993', '2018')).to_dataframe().corr(),
    #             annot=True)
    fig, ax = plt.subplots(figsize=(11, 11))

    im, cbar = heatmap(corr.values, corr.index.values, corr.columns, ax=ax,
                       cmap="YlGn", cbarlabel="correlation")
    texts = annotate_heatmap(im, valfmt="{x:.2f}")
    font = {'family': 'serif',
            'color': 'darkred',
            'weight': 'normal',
            'size': 16,
            }
    plt.title('Correlation Heatmap of times: {} to {} , latmean: {} to {}'.format(
            times[0], times[1], lats[0], lats[1]), fontdict=font)
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
    
