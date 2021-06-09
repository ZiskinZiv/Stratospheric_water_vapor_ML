#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 10:33:05 2018

@author: shlomi
"""

def convert_da_to_long_form_df(da, var_name=None, value_name=None):
    """ convert xarray dataarray to long form pandas df
    to use with seaborn"""
    import xarray as xr
    if var_name is None:
        var_name = 'var'
    if value_name is None:
        value_name = 'value'
    dims = [x for x in da.dims]
    if isinstance(da, xr.Dataset):
        value_vars = [x for x in da]
    elif isinstance(da, xr.DataArray):
        value_vars = [da.name]
    df = da.to_dataframe()
    for i, dim in enumerate(da.dims):
        df[dim] = df.index.get_level_values(i)
    df = df.melt(value_vars=value_vars, value_name=value_name,
                 id_vars=dims, var_name=var_name)
    return df


def anomalize_xr(da_ts, freq='D', time_dim=None, units=None, verbose=True):  # i.e., like deseason
    import xarray as xr
    if time_dim is None:
        time_dim = list(set(da_ts.dims))[0]
    attrs = da_ts.attrs
    if isinstance(da_ts, xr.Dataset):
        da_attrs = dict(zip([x for x in da_ts], [da_ts[x].attrs for x in da_ts]))
    try:
        name = da_ts.name
    except AttributeError:
        name = ''
    if isinstance(da_ts, xr.Dataset):
        name = [x for x in da_ts]
    if freq == 'D':
        if verbose:
            print('removing daily means from {}'.format(name))
        frq = 'daily'
        date = groupby_date_xr(da_ts)
        grp = date
    elif freq == 'H':
        if verbose:
            print('removing hourly means from {}'.format(name))
        frq = 'hourly'
        grp = '{}.hour'.format(time_dim)
    elif freq == 'MS':
        if verbose:
            print('removing monthly means from {}'.format(name))
        frq = 'monthly'
        grp = '{}.month'.format(time_dim)
    elif freq == 'AS':
        if verbose:
            print('removing yearly means from {}'.format(name))
        frq = 'yearly'
        grp = '{}.year'.format(time_dim)
    elif freq == 'DOY':
        if verbose:
            print('removing day of year means from {}'.format(name))
        frq = 'dayofyear'
        grp = '{}.dayofyear'.format(time_dim)
    elif freq == 'WOY':
        if verbose:
            print('removing week of year means from {}'.format(name))
        frq = 'weekofyear'
        grp = '{}.weekofyear'.format(time_dim)
    # calculate climatology:
    climatology = da_ts.groupby(grp).mean()
    climatology_std = da_ts.groupby(grp).std()
    da_anoms = da_ts.groupby(grp) - climatology
    if units == '%':
        da_anoms = 100.0 * (da_anoms.groupby(grp) / climatology)
        # da_anoms = 100.0 * (da_anoms / da_ts.mean())
        # da_anoms = 100.0 * (da_ts.groupby(grp)/climatology - 1)
        # da_anoms = 100.0 * (da_ts.groupby(grp)-climatology) / da_ts
        if verbose:
            print('Using % as units.')
    elif units == 'std':
        da_anoms = (da_anoms.groupby(grp) / climatology_std)
        if verbose:
            print('Using std as units.')
    da_anoms = da_anoms.reset_coords(drop=True)
    da_anoms.attrs.update(attrs)
    da_anoms.attrs.update(action='removed {} means'.format(frq))
    # if dataset, update attrs for each dataarray and add action='removed x means'
    if isinstance(da_ts, xr.Dataset):
        for x in da_ts:
            da_anoms[x].attrs.update(da_attrs.get(x))
            da_anoms[x].attrs.update(action='removed {} means'.format(frq))
            if units == '%':
                da_anoms[x].attrs.update(units='%')
    return da_anoms


def copy_coords_attrs(ds1, ds2, verbose=False):
    inter_coords = list(set(ds1.coords).intersection(set(ds2.coords)))
    for coord in inter_coords:
        if verbose:
            print('copying attrs for {} coord'.format(coord))
        for key, value in ds1[coord].attrs.items():
            ds2[coord].attrs[key] = value
    return ds2


def save_ncfile(xarray, savepath, filename='temp.nc', engine=None, dtype=None,
                fillvalue=None):
    import xarray as xr
    print('saving {} to {}'.format(filename, savepath))
    if dtype is None:
        comp = dict(zlib=True, complevel=9, _FillValue=fillvalue)  # best compression
    else:
        comp = dict(zlib=True, complevel=9, dtype=dtype, _FillValue=fillvalue)  # best compression
    if isinstance(xarray, xr.Dataset):
        encoding = {var: comp for var in xarray}
    elif isinstance(xarray, xr.DataArray):
        encoding = {var: comp for var in xarray.to_dataset()}
    xarray.to_netcdf(savepath / filename, 'w', encoding=encoding, engine=engine)
    print('File saved!')
    return


def query_yes_no(question, default="no"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    import sys
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")


def xr_reindex_with_date_range(ds, time_dim='time', drop=True, freq='MS'):
    import pandas as pd
    if drop:
        ds = ds.dropna(time_dim)
    start = pd.to_datetime(ds[time_dim].min().item())
    end = pd.to_datetime(ds[time_dim].max().item())
    new_time = pd.date_range(start, end, freq=freq)
    ds = ds.reindex({time_dim: new_time})
    return ds


def path_glob(path, glob_str='*.nc', return_empty_list=False):
    """returns all the files with full path(pathlib3 objs) if files exist in
    path, if not, returns FilenotFoundErro"""
    from pathlib import Path
#    if not isinstance(path, Path):
#        raise Exception('{} must be a pathlib object'.format(path))
    path = Path(path)
    files_with_path = [file for file in path.glob(glob_str) if file.is_file]
    if not files_with_path and not return_empty_list:
        raise FileNotFoundError('{} search in {} found no files.'.format(glob_str,
                        path))
    elif not files_with_path and return_empty_list:
        return files_with_path
    else:
        return files_with_path


def configure_logger(name='general', filename=None):
    import logging
    import sys
    stdout_handler = logging.StreamHandler(sys.stdout)
    if filename is not None:
        file_handler = logging.FileHandler(filename=filename, mode='a')
        handlers = [file_handler, stdout_handler]
    else:
        handlers = [stdout_handler]

    logging.basicConfig(
            level=logging.INFO,
            format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s',
            handlers=handlers
            )
    logger = logging.getLogger(name=name)
    return logger

# choice of picking all possible permutaions : 2**len(reg_vector) - 1


def adj_rsquare_xr(rs_da, n_samples, p_ind_vars):
    """produced ajusted rsquares score with n_samples,
    p_ind_vars(number of regressors) and rs_da dattarry of rsquared score"""
    # Adj r2 = 1-(1-R2)*(n-1)/(n-p-1)
    if not isinstance(n_samples, int):
        print('n_samples should be integer.')
        return
    if not isinstance(p_ind_vars, int):
        print('p_ind_vars should be integer.')
        return
    Adj_rsq = 1.0 - (1.0 - rs_da) * (n_samples - 1) /\
                    (n_samples - p_ind_vars - 1)
    return Adj_rsq


def reg_stack(regressors):
    import xarray as xr
    reg_da = xr.concat(regressors.data_vars, dim='regressors')
    reg_da['regressors'] = [x for x in regressors.data_vars.keys()]
    reg_da.name = 'R' + str(regressors.attrs['group_number'])
    return reg_da


def xr_weighted_mean(xarray, mean_on_lon=True, mean_on_lat=True):
    import xarray as xr
    from ML_OOP_stratosphere_gases import TargetArray
    attrs = xarray.attrs
    if isinstance(xarray, xr.DataArray):
        ds = xarray.to_dataset()
        was_da = True
    if isinstance(xarray, xr.Dataset) or isinstance(xarray, TargetArray):
        ds = xarray
        was_da = False
    ds = area_from_latlon_xr(ds)
    data_vars = [x for x in ds.data_vars.keys() if x != 'area']
    if 'lon' in ds.dims and mean_on_lon and mean_on_lat:
        # do mean on lon and lat:
        for var in data_vars:
            ds[var] = (ds['area'] * ds[var]).sum('lon') / ds['area'].sum('lon')
            ds[var] = (ds['area'].sum('lon') * ds[var]).sum('lat') / \
                ds['area'].sum('lon').sum('lat')
    elif 'lon' in ds.dims and not mean_on_lon and mean_on_lat:
        # do mean on lat only:
        for var in data_vars:
            ds[var] = (ds['area'].sum('lon') * ds[var]).sum('lat') / \
                ds['area'].sum('lon').sum('lat')
    elif 'lon' in ds.dims and mean_on_lon and not mean_on_lat:
        # do mean on lon only:
        for var in data_vars:
            ds[var] = (ds['area'] * ds[var]).sum('lon') / ds['area'].sum('lon')
    else:
        for var in data_vars:
            ds[var] = (ds['area'] * ds[var]).sum('lat') / ds['area'].sum('lat')
    if was_da:
        da = ds[data_vars].to_array(name=data_vars[0]).squeeze(drop=True)
        da = da.reset_coords(drop=True)
    else:
        da = ds[data_vars].reset_coords(drop=True)
    if 'time' in xarray.dims:
        da['time'] = xarray.time
    da.attrs = attrs
    da = xr_order(da)
    return da


def area_from_latlon_xr(ds, verbose=False):
    """input is data set with lat / lon coords. out put is area
    assuming regular evenly seperated lat/lon grid"""
    ds = xr_rename_sort(ds, lon_roll=True)
    if 'lat' in ds.dims:
        lat_attrs = ds.lat.attrs
        lat_res = max(ds.lat.diff('lat'))
        if 'lon' in ds.dims:
            lon_attrs = ds.lon.attrs
            lon_res = max(ds.lon.diff('lon'))
        else:
            if verbose:
                print('no lon dim in dataset...')
            lon_res = lat_res
        area_ds = grid_seperation_xr(lat_res, lon_res)
        area_ds = area_ds.rename({'lat_center': 'lat'})
        area_ds = area_ds.rename({'lon_center': 'lon'})
    else:
        if verbose:
            print('no lat dim in dataset...')
        return
    if 'lon' in ds.dims:
        ds['area'] = area_ds['area']
        ds.lon.attrs = lon_attrs
        if verbose:
            print('adding area for lat/lon coords.')
    else:
        ds['area'] = area_ds['area'].sum('lon')
        ds.lat.attrs = lat_attrs
        if verbose:
            print('adding area for lat coords.')
    return ds


def generic_regrid(datain, lats_in, lons_in, lats_out, lons_out, order,
                   long_second=True):
    """takes a geo gridded variable and regrids it to different grid"""
    # long_second is a flag that means the second dimension of datain is the longitude
    from mpl_toolkits import basemap
    import numpy as np
    if (not long_second):
        data_in = datain.T
    else:
        data_in = datain
    lons_tri = np.concatenate([lons_in - 360, lons_in, lons_in + 360])
    data_tri = np.concatenate([data_in, data_in, data_in], axis=1)
    lats, lons = np.meshgrid(lats_out, lons_out)
    # regrided_data = basemap.interp(data_tri, lats_in, lons_tri, lats, lons, order=order)
    regrided_data = basemap.interp(data_tri, lons_tri, lats_in, lons, lats,
                                   order=order).T
    return regrided_data


def grid_seperation_xr(lat_res=1.25, lon_res=1.25, area_equal=False,
                       clip_last_lon=True, lon_start=-180.0):
    import numpy as np
    import xarray as xr
    # calculates the real area of earth's(normalized to 1) at each latt+lon strip

    def calculate_area(lat_outer, lon_outer):
        area = np.zeros((len(lat_outer) - 1, len(lon_outer) - 1))
        lat_outer = np.deg2rad(lat_outer)
        lon_outer = np.deg2rad(lon_outer)
        for k in range(len(lat_outer) - 1):
            for j in range(len(lon_outer) - 1):
                area[k, j] = ((lon_outer[j + 1] - lon_outer[j]) / (4.0 * np.pi) *
                                (np.sin(lat_outer[k + 1]) - np.sin(lat_outer[k])))
        return area
    # create xarray of the area with the coords lat/lon etc...:

    def xarray_area(lat_outer, lon_outer, lat_center, lon_center,
                    area_equal, clip_last_lon):
        area = calculate_area(lat_outer, lon_outer)
        xarray = xr.Dataset()
        area = xr.DataArray(area, coords=[lat_center, lon_center], dims=['lat_center', 'lon_center'])
        xarray['area'] = area
        xarray['lat_outer'] = lat_outer
        if clip_last_lon:
            xarray['lon_outer'] = lon_outer[:-1]
        else:
            xarray['lon_outer'] = lon_outer
        if area_equal:
            xarray['area'].name = 'grid area for equal-area grid seperation'
        else:
            xarray['area'].name = 'grid area for nth by mth lat/lon grid seperation'
        return xarray

    n_size_lat = int(180.0 // lat_res)
    n_size_lon = int(360.0 // lon_res)
    double_lat = np.zeros((1, 2 * (n_size_lat) + 1)).squeeze()
    double_lon = np.zeros((1, 2 * (n_size_lon) + 1)).squeeze()
    lat_outer = np.zeros((1, n_size_lat + 1)).squeeze()
    lon_outer = np.zeros((1, n_size_lon + 1)).squeeze()
    lat_center = np.zeros((1, n_size_lat)).squeeze()
    lon_center = np.zeros((1, n_size_lon)).squeeze()
    # First calculate seperation to equal area zones
    lat_outer[0] = -np.pi / 2.0
    for i in np.arange(n_size_lat - 1) + 1:
        lat_outer[i] = np.arcsin(2.0 / n_size_lat + np.sin(lat_outer[i - 1]))
    lat_outer[n_size_lat] = np.pi / 2.0

    lon_outer[0] = np.deg2rad(lon_start)
    for i in np.arange(n_size_lon - 1) + 1:
        lon_outer[i] = lon_outer[i - 1] + 2.0 * np.pi / n_size_lon
    lon_outer[n_size_lon] = 2.0 * np.pi
    # mean latt where mean is the lattitude where the area is half between the previous calculated lines
    double_lat[0] = -np.pi / 2.0
    for i in np.arange(2 * n_size_lat - 1) + 1:
        double_lat[i] = np.arcsin(1.0 / n_size_lat + np.sin(double_lat[i - 1]))
    double_lat[2 * n_size_lat] = np.pi / 2.0
    for i in range(n_size_lat):
        lat_center[i] = double_lat[2 * i]
    double_lon[0] = 0.0
    for i in np.arange(2 * n_size_lon) + 1:
        double_lon[i] = double_lon[i - 1] + 1.0 * np.pi / n_size_lon
    for i in range(n_size_lon):
        lon_center[i] = double_lon[2 * i]
    lat_outer = np.rad2deg(lat_outer)
    lon_outer = np.rad2deg(lon_outer)
    lat_center = np.rad2deg(lat_center)
    lon_center = np.rad2deg(lon_center)
    equal_area_xr = xarray_area(lat_outer, lon_outer, lat_center, lon_center, True, True)
    # second calculate seperation and each area zone lat equal:
    lat_outer[0] = -90.0
    for i in np.arange(n_size_lat) + 1:
        lat_outer[i] = lat_outer[i - 1] + 180.0 * 1.0 / n_size_lat
    lon_outer[0] = lon_start
    for i in np.arange(n_size_lon) + 1:
        lon_outer[i] = lon_outer[i - 1] + 360.0 / n_size_lon
    for i in range(n_size_lat):
        lat_center[i] = (lat_outer[i] + lat_outer[i + 1]) / 2.0
    for i in range(n_size_lon):
        lon_center[i] = (lon_outer[i] + lon_outer[i + 1]) / 2.0
    regular_xr = xarray_area(lat_outer, lon_outer, lat_center, lon_center, False, True)
    if area_equal:
        return equal_area_xr
    else:
        return regular_xr


def xr_rename_sort(xarray, lon_roll=True, verbose=False):
    """sorts the lat lon dims from minus to plus and level to higher to lower
    if latitude and longitude renames to lat lon"""
    import xarray as xr

    def rename_da(da):
        # for data_arrays only:
        if 'latitude' in da.dims:
            da = da.rename({'latitude': 'lat'})
            if verbose:
                print('Fixing lat dim...')
        if 'longitude' in da.dims:
            da = da.rename({'longitude': 'lon'})
            if verbose:
                print('Fixing lon dim...')
        if 'levels' in da.dims:
            da = da.rename({'levels': 'level'})
            if verbose:
                print('Fixing level dim...')
        if 'time' in da.dims:
            da = da.sortby('time', ascending=True)
        if 'lat' in da.dims:
            da = da.sortby('lat', ascending=True)
        if 'level' in da.dims:
            da = da.sortby('level', ascending=False)
        if lon_roll:
            if 'lon' in da.dims:
                da = da.sortby('lon', ascending=True)
                if min(da['lon'].values) >= 0.0:
                    da = da.roll(lon=-180, roll_coords=False)
                    da['lon'] -= 180
                    if verbose:
                        print('Rolling lon dim to -180 to 180')
                else:
                    if verbose:
                        print('lon already rolled...')
            else:
                if verbose:
                    print('No lon dim found...')
        return da
    if type(xarray) == xr.DataArray:
        xarray = rename_da(xarray)
    elif type(xarray) == xr.Dataset:
        for da in xarray.data_vars.values():
            xarray[da.name] = rename_da(da)
    return xarray


def prepare_reg_list(reg_list):
    """turn the regressors list detasets into a list of datattys with regressors dim"""
    import xarray as xr
    new_reg_list = []
    for reg in reg_list:
        # names = [x for x in reg.data_vars if x != reg.attrs['median']]
        # data_reg = [x for x in reg.data_vars.values() if x != reg.attrs['median']]
        reg_dict = dict([(x, y) for x, y in reg.data_vars.items()
                        if x != reg.attrs['median']])
        con_data = xr.concat(reg_dict.values(), dim='regressors')
        con_data['regressors'] = list(reg_dict.keys())
        con_data.name = 'regressors_time_series'
        new_reg_list.append(con_data)
    return new_reg_list


def predict_xr(result_ds, regressors):
    """input: results_ds as came out of MLR and saved to file, regressors dataset"""
    # if produce_RI isn't called on data then you should explicitely put time info
    import xarray as xr
    import aux_functions_strat as aux
    rds = result_ds
    regressors = regressors.sel(time=rds.time)  # slice
    regressors = regressors.apply(aux.normalize_xr, norm=1, verbose=False)  # normalize
    reg_dict = dict(zip(rds.regressors.values, regressors.data_vars.values()))
    # make sure that all the regressors names are linking to their respective dataarrays
    for key, value in reg_dict.items():
        # print(key, value)
        assert value.name == key
    reg_da = xr.concat(reg_dict.values(), dim='regressors')
    reg_da['regressors'] = list(reg_dict.keys())
    reg_da.name = 'regressors_time_series'
    rds['predicted'] = xr.dot(rds.params, reg_da) + rds.intercept
    rds = aux.xr_order(rds)
    # retures the same dataset but with total predicted reconstructed geo-time-series field
    result_ds = rds
    return result_ds


def groupby_date_xr(da_ts, time_dim='time'):
    df = da_ts[time_dim].to_dataframe()
    df['date'] = df.index.date
    date = df['date'].to_xarray()
    return date


def custom_stack_xr(da, dim_not_stacked='time'):
    import xarray as xr
    import pandas as pd
    if type(da) != xr.DataArray:
        print('custom_stack accepts only xr.DataArray for now...')
        return
    dims_to_stack = [x for x in da.dims if dim_not_stacked != x]
    stacked_name = 'stacked'  # '_'.join(dims_to_stack) + '_stacked'
    stacked_da = da.stack(dumm=dims_to_stack)
    stacked_da = stacked_da.rename({'dumm': stacked_name})
    da_to_stack = [da[x] for x in da.dims if dim_not_stacked != x]
    stacked_dims = dict(zip(dims_to_stack, da_to_stack))
    mindex = pd.MultiIndex.from_product(stacked_dims.values(),
                                        names=(stacked_dims.keys()))
    return stacked_da, stacked_dims, mindex


def get_unique_index(da, dim='time', verbose=False):
    import numpy as np
    before = da[dim].size
    _, index = np.unique(da[dim], return_index=True)
    da = da.isel({dim: index})
    after = da[dim].size
    if verbose:
        print('dropped {} duplicate coord entries.'.format(before-after))
    return da


def xvar(da, dim):
    """accepts dataarray (xarray) and computes the variance across a dim"""
    temp = abs(da - da.mean(dim))**2
    return temp.mean(dim)


def text_white_underline(x, **kwargs):
    from termcolor import colored
    return colored(x, None, attrs=['bold', 'underline'], **kwargs)


def text_red(x, **kwargs):
    from termcolor import colored
    return colored(x, 'red', attrs=['bold'], **kwargs)


def text_blue(x, **kwargs):
    from termcolor import colored
    return colored(x, 'blue', attrs=['bold'], **kwargs)


def text_green(x, **kwargs):
    from termcolor import colored
    return colored(x, 'green', attrs=['bold'], **kwargs)


def text_yellow(x, **kwargs):
    from termcolor import colored
    return colored(x, 'yellow', attrs=['bold'], **kwargs)


def get_all_reg_combinations(dataset):
    """get all the possible combinations from a dataset's variables"""
    # return list of lists of datasets, first length is original dataset len,
    # second is the
    import itertools
    import xarray as xr
    if type(dataset) != xr.Dataset:
        return print('Input is xarray dataset only')
    else:
        com_list = []
        for i in range(len(dataset.data_vars)):
            com_list.append(list(itertools.combinations(dataset.data_vars.values(), i+1)))
        ds_list = []
        for com_i in range(len(com_list)):
            for ds_num in range(len(com_list[com_i])):
                ds_list.append(xr.merge(com_list[com_i][ds_num]))
    return ds_list


def get_RI_reg_combinations(dataset):
    """return n+1 sized dataset of full regressors and median value regressors"""
    import xarray as xr

    def replace_dta_with_median(dataset, dta):
        ds = dataset.copy()
        ds[dta] = dataset[dta] - dataset[dta] + dataset[dta].median('time')
        ds.attrs['median'] = dta
        return ds
    if type(dataset) != xr.Dataset:
        return print('Input is xarray dataset only')
    ds_list = []
    ds_list.append(dataset)
    dataset.attrs['median'] = 'full_set'
    for da in dataset.data_vars:
        ds_list.append(replace_dta_with_median(dataset, da))
    return ds_list


def overlap_time_xr(*args, time_dim='time'):
    """return the intersection of datetime objects from time field in *args"""
    # caution: for each arg input is xarray with dim:time
    time_list = []
    try:
        for ts in args:
            time_list.append(ts[time_dim].values)
    except KeyError:
        print('"{}" dim should be at all args...'.format(time_dim))
    intersection = set.intersection(*map(set, time_list))
    intr = sorted(list(intersection))
    return intr


def dim_intersection(da_list, dim='time', dropna=True, verbose=None):
    import pandas as pd
    if dropna:
        setlist = [set(x.dropna(dim)[dim].values) for x in da_list]
    else:
        setlist = [set(x[dim].values) for x in da_list]
    empty_list = [x for x in setlist if not x]
    if empty_list:
        if verbose == 0:
            print('NaN dim drop detected, check da...')
        return None
    u = list(set.intersection(*setlist))
    # new_dim = list(set(a.dropna(dim)[dim].values).intersection(
    #     set(b.dropna(dim)[dim].values)))
    if dim == 'time':
        new_dim = sorted(pd.to_datetime(u))
    else:
        new_dim = sorted(u)
    return new_dim


def xr_order(xrr, dims_order=['time', 'level', 'lat', 'lon']):
    """order the dims of xarray dataarray acorrding to some rules"""
    import xarray as xr

    def order_da(da, dims_order):
        # get list of dims in da:
        da_dims = list(da.dims)
        # get list of dims other than specified in dims_order:
        other_dims = list((set(da_dims) | set(dims_order)) - set(dims_order))
        # concatenate them:
        wanted_dims_order = dims_order + other_dims
        # wanted_dict = {d.index(dim): dim for dim in d}
        dims_dict = {}
        # build a dict that the key is the position of the ordered dims and the
        # value is da_dims if the dims name match (if some or all the dims in dims_order)
        for dim in da_dims:
            i = [wanted_dims_order.index(x) for x in wanted_dims_order if x == dim]
            dims_dict[i[0]-1] = dim
        # sort the keys:
        sorted_keys = sorted(dims_dict.keys())
        # build a list of the sorted dims:
        dims_list = [dims_dict[key] for key in sorted_keys]
        # transpose them and voila:
        da = da.transpose(*dims_list)
        return da
    # interface allows for dataarrays as well as datasets:
    if type(xrr) == xr.Dataset:
        for da_name in xrr.data_vars.keys():
            xrr[da_name] = order_da(xrr[da_name], dims_order)
    elif type(xrr) == xr.DataArray:
        xrr = order_da(xrr, dims_order)
    return xrr


def weighted_mean_decraped(xarray):
    """take xarray and return xarray with lat lon dim gone all weighted"""
    # weights are only cos(lat)
    import xarray as xr
    import numpy as np
    if type(xarray) != xr.DataArray:
        print('input is only DataArray (xarray), sorry...')
        return
    xarray = xr_rename_sort(xarray)
    xarray = xr_order(xarray)  # order the dims like this: time,level,lat,lon or time,lat,lon or time,lat or lat
    # get lat dim and coords:
    lat = xarray.lat.values
    # mask Infs and NaNs:
    m_xr = np.ma.masked_invalid(xarray.values)
    # first check if data is zonal mean:
    if 'lon' in xarray.coords.keys():
        # do mean to get field in each lat (zonal mean)
        # mean_data_lat = xarray.mean('lon')
        mean_data_lat = np.nanmean(m_xr, axis=xarray.dims.index('lon'))
    else:
        mean_data_lat = m_xr
    # Find Weighted mean of those values
    # num = np.nansum(np.cos(np.deg2rad(xarray.lat.values)) * mean_data_lat.values, axis=2)
    num = np.nansum(np.sqrt(np.cos(np.deg2rad(lat))) * mean_data_lat, axis=xarray.dims.index('lat'))
    # denom = np.nansum(np.cos(np.deg2rad(xarray.lat.values)))
    denom = np.nansum(np.sqrt(np.cos(np.deg2rad(lat))))
    # Find mean global temperature
    mean_global = num / denom
    # construct an xarray, and rename dim1 to level
    if 'level' in xarray.dims:
        xr_w = xr.DataArray(mean_global, coords=[xarray.time, xarray.level],
                            dims=['time', 'level'])
    else:
        xr_w = xr.DataArray(mean_global, coords=[xarray.time], dims=['time'])
    xr_w.attrs = xarray.attrs
    xr_w.name = xarray.name
    return xr_w


def deseason_xr(data, how='std', month='all', season=None, clim=False,
                verbose=True, tdim='time'):
    """deseason data: 'mean'= remove long-term monthly mean
    'std'= remove long-term monthly mean and divide by long term monthly std
    month is selecting the spesific months, season is selecting the spesific seasons,
    clim is returnin only the climatology"""
    # clim is if you want to return only the climatology i.e 12 bins of data
    import xarray as xr
    if type(data) != xr.DataArray:  # support just for single field for now
        print('input is only DataArray (xarray), sorry...')
        return
    attrs = data.attrs
    field_name = data.name
    # first compute the monthly and seasonaly long term mean and std:
    month_mean = data.groupby(tdim + '.month').mean(tdim, keep_attrs=True)
    month_std = data.groupby(tdim + '.month').std(tdim, keep_attrs=True)
    season_mean = data.groupby(tdim + '.season').mean(tdim, keep_attrs=True)
    season_std = data.groupby(tdim + '.season').std(tdim, keep_attrs=True)
    if (month == 'all') and (season == 'all'):
        return print('month and season cant be both all, pick one..')
    # first do all data (all months):
    if (month == 'all') and (season is None):
        # check for climatology flag, if true return Dataset
        if clim:
            xds = xr.Dataset()
            xds['month_mean'] = month_mean
            xds['month_std'] = month_std
            xds.attrs = attrs
            xds.attrs['title'] = 'climatology'
            if verbose:
                print('returning monthly climatology dataset...')
            return xds
        elif how == 'mean':
            # remove just the monthly mean:
            xrr = data.groupby(tdim + '.month') - month_mean
            comment = 'removed the long term monthly mean'
        elif how == 'std':
            # remove the monthly mean and divide by monthly std
            data = data.groupby(tdim + '.month') - month_mean
            xrr = data.groupby(tdim + '.month') / month_std
            comment = 'removed the long term monthly mean and divided by the long trem monthly std'
    # now check for seasonal mean only:
    elif season == 'all':
        # check for climatology flag, if true return Dataset
        if clim:
            xds = xr.Dataset()
            xds['season_mean'] = season_mean
            xds['season_std'] = season_std
            xds.attrs = attrs
            xds.attrs['title'] = 'climatology'
            if verbose:
                print('returning seasonal climatology dataset...')
            return xds
        if how == 'mean':
            # remove just the seasonal mean:
            xrr = data.groupby('time.season') - season_mean
            comment = 'removed the seasonal mean'
        elif how == 'std':
            # remove the seasonal mean and divide by seasonal std
            data = data.groupby(tdim + '.season') - season_mean
            xrr = data.groupby(tdim + '.season') / season_std
            comment = 'removed the seasonal mean and divided by the seasonal std'
    # now slice only spesific month from data:
    elif (month != 'all') and (season is None):
        data = data.sel(time=data[tdim + '.month'] == month)
        if how == 'mean':
            # remove just the spesific month mean:
            xrr = data - data.mean(tdim)
            comment = 'selected month #' + str(month) + ' and removed the time mean'
        elif how == 'std':
            # remove the spesific month mean and divide by month std
            data = data - data.mean(tdim)
            xrr = data / data.std(tdim)
            comment = 'selected month #' + str(month) + ' and removed the time mean and divided by std'
    elif (season != 'all') and (season is not None):
        data = data.sel(time=data[tdim + '.season'] == season)
        if how == 'mean':
            # remove just the spesific season mean:
            xrr = data - data.mean(tdim)
            comment = 'selected season ' + str(season) + ' and removed the time mean'
        elif how == 'std':
            # remove the spesific season mean and divide by season std
            data = data - data.mean(tdim)
            xrr = data / data.std(tdim)
            comment = 'selected season ' + str(season) + ' and removed the time mean and divided by std'
    if verbose:
        if field_name is not None:
            print(comment + ' on ' + field_name)
    xrr.attrs = attrs
    xrr.attrs['comment'] = comment
    xrr.name = field_name
    xrr = xrr.reset_coords(drop=True)
    return xrr


def normalize_xr(data, norm=1, down_bound=-1., upper_bound=1., verbose=True):
    attrs = data.attrs
    avg = data.mean('time', keep_attrs=True)
    sd = data.std('time', keep_attrs=True)
    try:
        data_name = data.name
    except AttributeError:
        try:
            data_name = ', '.join([x for x in data.data_vars])
        except:
            data_name = ''
    if norm == 0:
        data = data
        norm_str = 'No'
    elif norm == 1:
        data = (data-avg)/sd
        norm_str = '(data-avg)/std'
    elif norm == 2:
        data = (data-avg)/avg
        norm_str = '(data-avg)/avg'
    elif norm == 3:
        data = data/avg
        norm_str = '(data/avg)'
    elif norm == 4:
        data = data/sd
        norm_str = '(data)/std'
    elif norm == 5:
        dh = data.max()
        dl = data.min()
        # print dl
        data = (((data-dl)*(upper_bound-down_bound))/(dh-dl))+down_bound
        norm_str = 'mapped between ' + str(down_bound) + ' and ' + str(upper_bound)
        # print data
        if verbose:
            print('Data is {} on {}'.format(norm_str, data_name))
    elif norm == 6:
        data = data-avg
        norm_str = 'data-avg'
    if verbose and norm != 5:
        print('Preforming {} Normalization on {}'.format(norm_str, data_name))
    data.attrs = attrs
    data.attrs['Normalize'] = norm_str
    return data


def remove_nan_xr(data, just_geo=True, verbose=False):
    """try to remove nans with dropping each label at a spesific dim xarry"""
    from itertools import permutations
    import xarray as xr
    import aux_functions_strat as aux
    if type(data) != xr.DataArray:
        print('Data is not xarray.DataArray...')
        return
    else:
        if just_geo:
            dims_list = [x for x in data.dims if x != 'time']
        else:
            dims_list = data.dims
        perm_list = list(permutations(dims_list, len(dims_list)))
        for dims in perm_list:
            data_copy = data.copy()
            for dim in dims:
                data_copy = data_copy.dropna(dim)
                non_nans = aux.desc_nan(data_copy, verbose=False)
                # print(non_nans)
            if non_nans != 0:
                if verbose:
                    print('found good combination nan removal: ' + str(dims))
                return data_copy
        aux.text_red('removing nans with dim labels failed...try something else...')
    return


#def xr_transpose_dim_to_last_pos(da, last_dim='lat'):
#    """transpose an xarray dim to the last position so certain operations,
#    like matrix mul can be done"""
#    dims = [x for x in da.dims]
#    dim_idx = dims.index(last_dim)
#    if dim_idx != len(dims) - 1:
#        # check if not already last place
#        dims[dim_idx], dims[-1] = dims[-1], dims[dim_idx]
#        da = da.transpose(*dims)
#    return da


def lat_mean(xarray, method='cos', dim='lat', copy_attrs=True):
    import numpy as np
    import xarray as xr

    def mean_single_da(da, dim=dim, method=method):
        if dim not in da.dims:
            return da
        if method == 'cos':
            weights = np.cos(np.deg2rad(da[dim].values))
            da_mean = (weights * da).sum(dim) / sum(weights)
        if copy_attrs:
            da_mean.attrs = da.attrs
        return da_mean

    xarray = xarray.transpose(..., dim)
    if isinstance(xarray, xr.DataArray):
        xarray = mean_single_da(xarray)
    elif isinstance(xarray, xr.Dataset):
        xarray = xarray.map(mean_single_da, keep_attrs=copy_attrs)
    return xarray


def desc_nan(data, verbose=True):
    """count only NaNs in data and returns the thier amount and the non-NaNs"""
    import numpy as np
    import xarray as xr

    def nan_da(data):
        nans = np.count_nonzero(np.isnan(data.values))
        non_nans = np.count_nonzero(~np.isnan(data.values))
        if verbose:
            print(str(type(data)))
            print(data.name + ': non-NaN entries: ' + str(non_nans) + ' of total ' +
                  str(data.size) + ', shape:' + str(data.shape) + ', type:' +
                  str(data.dtype))
            print('Dimensions:')
        dim_nn_list = []
        for dim in data.dims:
            dim_len = data[dim].size
            dim_non_nans = np.int(data.dropna(dim)[dim].count())
            dim_nn_list.append(dim_non_nans)
            if verbose:
                print(dim + ': non-NaN labels: ' + str(dim_non_nans) + ' of total ' +
                      str(dim_len))
        return non_nans
    if type(data) == xr.DataArray:
        nn_dict = nan_da(data)
        return nn_dict
    elif type(data) == np.ndarray:
        nans = np.count_nonzero(np.isnan(data))
        non_nans = np.count_nonzero(~np.isnan(data))
        if verbose:
            print(str(type(data)))
            print('non-NaN entries: ' + str(non_nans) + ' of total ' +
                  str(data.size) + ', shape:' + str(data.shape) + ', type:' +
                  str(data.dtype))
    elif type(data) == xr.Dataset:
        for varname in data.data_vars.keys():
            non_nans = nan_da(data[varname])
    return non_nans


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
