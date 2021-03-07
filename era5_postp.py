#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 11:03:39 2019

@author: shlomi
"""


def concat_era5T(ds):
    import xarray as xr
    field_0001 = [x for x in ds.data_vars if '0001' in x]
    field_0005 = [x for x in ds.data_vars if '0005' in x]
    field = [x for x in ds.data_vars if '0001' not in x and '0005' not in x]
    if field_0001 and field_0005:
        da = xr.concat([ds[field_0001[0]].dropna('time'),
                        ds[field_0005[0]].dropna('time')], 'time')
        da.name = field_0001[0].split('_')[0]
    elif not field_0001 and not field_0005:
        return ds
    if field:
        da = xr.concat([ds[field[0]].dropna('time'), da], 'time')
    dss = da.to_dataset(name=field[0])
    return dss


def proc_era5(path, field, model_name):
    import xarray as xr
    import numpy as np
    from aux_functions_strat import path_glob
    from aux_functions_strat import save_ncfile
    files = sorted(path_glob(path, 'era5_{}_*.nc'.format(field)))
    if 'single' in model_name:
        era5 = xr.open_mfdataset(files)
        era5 = concat_era5T(era5)
        start = era5.time.dt.year[0].values.item()
        end = era5.time.dt.year[-1].values.item()
        filename = '_'.join(['era5', str(field), '4Xdaily', str(start) +
                             '-' + str(end)])
        filename += '.nc'
        save_ncfile(era5, path, filename)
        print('Done!')
    elif 'pressure' in model_name:
        era5 = xr.open_mfdataset(files)
        era5 = concat_era5T(era5)
        start = era5.time.dt.year[0].values.item()
        end = era5.time.dt.year[-1].values.item()
        years = np.arange(start, end + 1).tolist()
        for year in years:
            era5_yearly = era5.sel(time=str(year))
            filename = '_'.join(['era5', str(field), '4Xdaily', str(year)])
            filename += '.nc'
            save_ncfile(era5_yearly, path, filename)
        print('Done!')
    return


def check_path(path):
    import os
    from pathlib import Path
    path = str(path)
    if not os.path.exists(path):
        raise argparse.ArgumentTypeError(path + ' does not exist...')
    return Path(path)


if __name__ == '__main__':
    import argparse
    import sys
    from cds_era5_script import era5_variable
    era5_var = era5_variable()
    era5_var.start_year = 1979
    era5_var.end_year = 2019
    parser = argparse.ArgumentParser(description='Era5 post-proccessor')
    optional = parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    # remove this line: optional = parser...
    required.add_argument('--path', help="a full path to save in the cluster,\
                          e.g., /data11/ziskin/", type=check_path)
    required.add_argument('--field', help="era5 field abbreviation, e.g., T,\
                          U , V", type=str, choices=era5_var.list_vars(),
                          metavar='Era5 Field name abbreviation')
#    optional.add_argument('--type', help='Use single for single level products,' + 
#                          ' pressure for pressure level products.'
#                          , type=str, choices=['single', 'pressure'])
#                          metavar=str(cds.start_year) + ' to ' + str(cds.end_year))
#    optional.add_argument('--half', help='a spescific six months to download,\
#                          e.g, 1 or 2', type=int, choices=[1, 2],
#                          metavar='1 or 2')
#    parser._action_groups.append(optional)  # added this line
    args = parser.parse_args()
    # print(parser.format_help())
#    # print(vars(args))
    if args.path is None:
        print('path is a required argument, run with -h...')
        sys.exit()
    elif args.field is None:
        print('field is a required argument, run with -h...')
        sys.exit()
    cds_obj = era5_var.get_model_name(args.field)
    print('post-proccessing era5 all years, field: ' + args.field +
          ', saving to path:' + args.path.as_posix())
    proc_era5(args.path, args.field, era5_var.model_name)
