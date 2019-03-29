#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 13:43:35 2018
This script is for proccesing SWOOSH data (assimalated water vapor mixing ratio
for the stratosphere from different satallite platforms)
@author: shlomi
"""

def get_swoosh_xarray(filename='swoosh-v02.6-198401-201802-latpress-5deg-L31.nc'):
    """Takes files in the path and outputs the combined_h20/o3_q data and the
    lat, lon, level etc..."""
    import os, os.path
    import sys
    import xarray as xr
    import pandas as pd
    if sys.platform == 'linux':
        path = '/home/shlomi/Desktop/DATA/Work Files/Chaim_Stratosphere_Data/'
    elif sys.platform == 'darwin':  # mac os
        path = '/Users/shlomi/Documents/Chaim_Stratosphere_Data/'
    if os.path.isfile(os.path.join(path, filename)):
        xsw = xr.open_dataset(path + filename, decode_times=False)
        print('Importing ' + path + filename + ' to Xarray')
    else:
        print('File not found...')
    k = list(xsw.data_vars.keys())  # list data all variables
    combined_list=[key for key in xsw.keys() if 'combined' in key.lower()] # list all vars with combined in them
    # choose which data variables to drop:
    to_drop = list(set(k) - set(combined_list))
    # k.remove('month')
#    k.remove('year')
#    # k.remove('yrtime')
#    # k.remove('jultime')
#    k.remove('combinedanomh2oq')
#    k.remove('combinedh2oq')
#    k.remove('combinedanomo3q')
#    k.remove('combinedo3q')
#    k.remove('combinedseash2oq')
#    k.remove('combinedseaso3q')
    # remove all other data vars:
    xsw = xsw.drop(to_drop)
    # drop last 3 data in data vars and in time (negative year):
    # xsw = xsw.where((xsw.year > 0), drop=True)
    # parse date time
    time = pd.date_range('1984-01-01', freq='MS', periods=len(xsw.time))
    xsw['time'] = time
    # xsw = xsw.drop('year')
    return xsw


def interp_sw_levels_xr(sw_xr_var):
    """takes swoosh data xarray Dataarray (one var) and interpolate it 
    vertically to MERRA GCM pressure levels"""
    # *levels is other optional pressure levels
    from scipy.interpolate import interp1d
    from scipy.interpolate import UnivariateSpline
    import numpy as np
    import xarray as xr
    MERRA_levels = np.array([3.00000000e+02,   2.50000000e+02,   2.00000000e+02,
                             1.50000000e+02,   1.00000000e+02,   7.00000000e+01,
                             5.00000000e+01,   4.00000000e+01,   3.00000000e+01,
                             2.00000000e+01,   1.00000000e+01,   7.00000000e+00,
                             5.00000000e+00,   4.00000000e+00,   3.00000000e+00,
                             2.00000000e+00,   1.00000000e+00])
    # logMlevels = np.flip(np.log(MERRA_levels), 0)
    logMlevels = np.log(MERRA_levels)
    levels = sw_xr_var.level.values
    sw_lon = sw_xr_var.lon.values
    sw_lat = sw_xr_var.lat.values
    # loglevels = np.flip(np.log(levels), 0)
    loglevels = np.log(levels)
    var = sw_xr_var.values
    var_out = np.empty((var.shape[0], len(MERRA_levels), var.shape[-2],
                        var.shape[-1]), dtype=var.dtype)

    for time in range(var.shape[0]):
        print(time)
        for lon in range(len(sw_lon)):
            for lat in range(len(sw_lat)):  
                # y = np.flip(var[time, :, lat, lon].squeeze(), 0)
                y = var[time, :, lat, lon].squeeze()
                # w = np.isnan(y)
                # y[w] = 0
                # f = UnivariateSpline(loglevels, y, w=~w, ext=1, check_finite=False)
                # var_out[time, :, lat, lon] = np.flip(f(logMlevels), 0)
                # var_out = interp_along_axis(y, loglevels, logMlevels, axis=0, inverse=True)
                
    var_out[var_out <= 0] = 0
    xr_out = xr.DataArray(var_out, coords=[sw_xr_var.time, MERRA_levels,
                                           sw_xr_var.lat, sw_xr_var.lon],
                                           dims=['time', 'level', 'lat', 'lon'])
    xr_out.attrs = sw_xr_var.attrs
    xr_out.name = sw_xr_var.name
    return xr_out


def interp_sw_xr_easy(sw_xr_var):
    """takes swoosh data xarray Dataarray (one var) and interpolate it 
    vertically to MERRA GCM pressure levels the easy way"""
    import numpy as np
    MERRA_levels = np.array([3.00000000e+02,   2.50000000e+02,   2.00000000e+02,
                             1.50000000e+02,   1.00000000e+02,   7.00000000e+01,
                             5.00000000e+01,   4.00000000e+01,   3.00000000e+01,
                             2.00000000e+01,   1.00000000e+01,   7.00000000e+00,
                             5.00000000e+00,   4.00000000e+00,   3.00000000e+00,
                             2.00000000e+00,   1.00000000e+00])
    # linear interpolation
    xr_out = sw_xr_var.interp(coords={'level':MERRA_levels}, method='linear')
    return xr_out
    
def prepare_swoosh_xr(swoosh_filename='swoosh-v02.6-198401-201802-lonlatpress-20deg-5deg-L31.nc', 
                      output_file='swoosh_interpolated_xr.nc'):
    import os, os.path
    import sys
    import xarray as xr
    if sys.platform == 'linux':
        path = '/home/shlomi/Desktop/DATA/Work Files/Chaim_Stratosphere_Data/'
    elif sys.platform == 'darwin':  # mac os
        path = '/Users/shlomi/Documents/Chaim_Stratosphere_Data/'
    if os.path.isfile(os.path.join(path, output_file)):
        print('filename already exists...delete or pick another name')
        return
    else:
        print('filename is good, carrying on...')
    sw = get_swoosh_xarray()
    varlist = [name for name in sw.data_vars.keys() if 'combined' in name.lower()]
    datarr = []
    for i in range(len(varlist)):
        xr_to_add = interp_sw_levels_xr(sw[varlist[i]])
        xrr = xr_to_add.to_dataset(name=varlist[i])
        datarr.append(xrr)
    sw_out = xr.merge(datarr)
    sw_out.attrs['long_name'] = 'Interpolated level from MERRA GCM levels, data from swoosh file: ' + swoosh_filename 
    sw_out.to_netcdf(path + output_file)
    print('nc file written to ' + path + output_file)
    return # sw_out


def get_all_swoosh_files(path):
    import os
    import xarray as xr
    import fnmatch
    import pandas as pd
    """Reads all swoosh dataset downloaded from swoosh website and write
    them as xarray netcdf files, basically adding datetime index and importing
    just the 'combined' fields."""
    filenames = []
    for file in os.listdir(path + '.'):
        if fnmatch.fnmatch(file, 'swoosh-v02.6*.*'):
            filenames.append(file)
    datasets = []
    for sw_filename in filenames:
        dataset = xr.open_dataset(path + sw_filename, decode_times=False)
        dataset.attrs['filename'] = sw_filename
        datasets.append(dataset)
        print('importing ' + sw_filename)
    data_dict = {}
    for i in range(len(datasets)):
        k = list(datasets[i].data_vars.keys())
        combined_list = [name for name in datasets[i].data_vars.keys()
                         if 'combined' in name.lower()]
        to_drop = list(set(k) - set(combined_list))
        datasets[i] = datasets[i].drop(to_drop)
        time = pd.date_range('1984-01-01', freq='MS',
                             periods=len(datasets[i].time))
        datasets[i]['time'] = time
        if 'latpress' in datasets[i].attrs['filename'].split('-'):
            dname = '-'.join(datasets[i].attrs['filename'].split('-')[-3:-1])
        elif 'lonlatpress' in datasets[i].attrs['filename'].split('-'):
            dname = '-'.join(datasets[i].attrs['filename'].split('-')[-4:-1])
        elif 'lattheta' in datasets[i].attrs['filename'].split('-'):
            dname = '-'.join(datasets[i].attrs['filename'].split('-')[-3:-1])
        data_dict[dname] = datasets[i]
        data_dict[dname].to_netcdf(path + 'swoosh_' + dname + '.nc')
        print('Saved ' + dname + ' to nc file, in ' + path)
    return data_dict


def compare_all_swoosh(sw_d):
    import matplotlib.pyplot as plt
    # need to run main_alaysis.py
    # sw_d = get_all_swoosh_files()
    for sw in sw_d:
        plot_pressure_time_xr(sw['combinedh2oq'].sel(lat=slice(-20, 20)))
        title = 'combinedh2oq_' + '_'.join(sw.attrs['filename'].split('-')[4:])
        title = title.replace('.nc', '')
        ax = plt.gca()
        ax.set_title(title)
        plt.savefig(title + '.pdf')
    return
                          

def interp_along_axis(y, x, newx, axis, inverse=False, method='linear'):
    """ Interpolate vertical profiles, e.g. of atmospheric variables
    using vectorized numpy operations

    This function assumes that the x-xoordinate increases monotonically

    ps:
    * Updated to work with irregularly spaced x-coordinate.
    * Updated to work with irregularly spaced newx-coordinate
    * Updated to easily inverse the direction of the x-coordinate
    * Updated to fill with nans outside extrapolation range
    * Updated to include a linear interpolation method as well
        (it was initially written for a cubic function)

    Peter Kalverla
    March 2018

    --------------------
    More info:
    Algorithm from: http://www.paulinternet.nl/?page=bicubic
    It approximates y = f(x) = ax^3 + bx^2 + cx + d
    where y may be an ndarray input vector
    Returns f(newx)

    The algorithm uses the derivative f'(x) = 3ax^2 + 2bx + c
    and uses the fact that:
    f(0) = d
    f(1) = a + b + c + d
    f'(0) = c
    f'(1) = 3a + 2b + c

    Rewriting this yields expressions for a, b, c, d:
    a = 2f(0) - 2f(1) + f'(0) + f'(1)
    b = -3f(0) + 3f(1) - 2f'(0) - f'(1)
    c = f'(0)
    d = f(0)

    These can be evaluated at two neighbouring points in x and
    as such constitute the piecewise cubic interpolator.
    """
    import numpy as np
    import warnings

    # View of x and y with axis as first dimension
    if inverse:
        _x = np.moveaxis(x, axis, 0)[::-1, ...]
        _y = np.moveaxis(y, axis, 0)[::-1, ...]
        _newx = np.moveaxis(newx, axis, 0)[::-1, ...]
    else:
        _y = np.moveaxis(y, axis, 0)
        _x = np.moveaxis(x, axis, 0)
        _newx = np.moveaxis(newx, axis, 0)

    # Sanity checks
    if np.any(_newx[0] < _x[0]) or np.any(_newx[-1] > _x[-1]):
        # raise ValueError('This function cannot extrapolate')
        warnings.warn("Some values are outside the interpolation range. "
                      "These will be filled with NaN")
    if np.any(np.diff(_x, axis=0) < 0):
        raise ValueError('x should increase monotonically')
    if np.any(np.diff(_newx, axis=0) < 0):
        raise ValueError('newx should increase monotonically')

    # Cubic interpolation needs the gradient of y in addition to its values
    if method == 'cubic':
        # For now, simply use a numpy function to get the derivatives
        # This produces the largest memory overhead of the function and
        # could alternatively be done in passing.
        ydx = np.gradient(_y, axis=0, edge_order=2)

    # This will later be concatenated with a dynamic '0th' index
    ind = [i for i in np.indices(_y.shape[1:])]

    # Allocate the output array
    original_dims = _y.shape
    newdims = list(original_dims)
    newdims[0] = len(_newx)
    newy = np.zeros(newdims)

    # set initial bounds
    i_lower = np.zeros(_x.shape[1:], dtype=int)
    i_upper = np.ones(_x.shape[1:], dtype=int)
    x_lower = _x[0, ...]
    x_upper = _x[1, ...]

    for i, xi in enumerate(_newx):
        # Start at the 'bottom' of the array and work upwards
        # This only works if x and newx increase monotonically

        # Update bounds where necessary and possible
        needs_update = (xi > x_upper) & (i_upper+1<len(_x))
        # print x_upper.max(), np.any(needs_update)
        while np.any(needs_update):
            i_lower = np.where(needs_update, i_lower+1, i_lower)
            i_upper = i_lower + 1
            x_lower = _x[[i_lower]+ind]
            x_upper = _x[[i_upper]+ind]

            # Check again
            needs_update = (xi > x_upper) & (i_upper+1<len(_x))

        # Express the position of xi relative to its neighbours
        xj = (xi-x_lower)/(x_upper - x_lower)

        # Determine where there is a valid interpolation range
        within_bounds = (_x[0, ...] < xi) & (xi < _x[-1, ...])

        if method == 'linear':
            f0, f1 = _y[[i_lower]+ind], _y[[i_upper]+ind]
            a = f1 - f0
            b = f0

            newy[i, ...] = np.where(within_bounds, a*xj+b, np.nan)

        elif method=='cubic':
            f0, f1 = _y[[i_lower]+ind], _y[[i_upper]+ind]
            df0, df1 = ydx[[i_lower]+ind], ydx[[i_upper]+ind]

            a = 2*f0 - 2*f1 + df0 + df1
            b = -3*f0 + 3*f1 - 2*df0 - df1
            c = df0
            d = f0

            newy[i, ...] = np.where(within_bounds, a*xj**3 + b*xj**2 + c*xj + d, np.nan)

        else:
            raise ValueError("invalid interpolation method"
                             "(choose 'linear' or 'cubic')")

    if inverse:
        newy = newy[::-1, ...]

    return np.moveaxis(newy, 0, axis)
